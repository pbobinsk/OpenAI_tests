import numpy as np
import openai
from pypdf import PdfReader
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

import redis

INDEX_NAME = "embeddings-index"           # Nazwa indeksu wyszukiwania
PREFIX = "doc"                            # Prefiks kluczy dokumentu
# Metryka odległości między wektorami (np. COSINE, IP, L2)
DISTANCE_METRIC = "COSINE"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "" 

class DataService():

    def __init__(self):
        # Połączenie z bazą Redis.
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD
        )

    def drop_redis_data(self, index_name: str = INDEX_NAME):
        try:
            self.redis_client.ft(index_name).dropindex()
            print('Indeks usunięty')
        except:
            # Indeks nie istnieje
            print('Indeks nie istnieje')

    def load_data_to_redis(self, embeddings):
        # Stałe
        vector_dim = len(embeddings[0]['vector'])  # Długość wektora
        
		    # Początkowa liczba wektorów
        vector_number = len(embeddings)

        # Definicja pól RediSearch
        text = TextField(name="text")
        text_embedding = VectorField("vector",
                                     "FLAT", {
                                         "TYPE": "FLOAT32",
                                         "DIM": vector_dim,
                                         "DISTANCE_METRIC": "COSINE",
                                         "INITIAL_CAP": vector_number,
                                     }
                                     )
        fields = [text, text_embedding]

        # Sprawdzenie, czy indeks istnieje
        try:
            self.redis_client.ft(INDEX_NAME).info()
            print("Indeks istnieje")
        except:
            # Utworzenie indeksu RediSearch
            self.redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(
                    prefix=[PREFIX], index_type=IndexType.HASH)
            )

        for embedding in embeddings:
            key = f"{PREFIX}:{str(embedding['id'])}"
            embedding["vector"] = np.array(
                embedding["vector"], dtype=np.float32).tobytes()
            self.redis_client.hset(key, mapping=embedding)
        print(
            f"Załadowano {self.redis_client.info()['db0']['keys']} dokumentów do indeksu wyszukiwania Redis o nazwie {INDEX_NAME}")

    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int = 1000):
        # Odczytanie pliku PDF i podzielenie danych na fragmenty
        reader = PdfReader(pdf_path)
        chunks = []
        for page in reader.pages:
            text_page = page.extract_text()
            chunks.extend([text_page[i:i+chunk_length].replace('\n', '')
                          for i in range(0, len(text_page), chunk_length)])

        # Utworzenie osadzeń
        response = openai.Embedding.create(
            model='text-embedding-ada-002', input=chunks)
        return [{'id': value['index'], 'vector':value['embedding'], 'text':chunks[value['index']]} for value in response['data']]

    def search_redis(self,
                     user_query: str,
                     index_name: str = "embeddings-index",
                     vector_field: str = "vector",
                     return_fields: list = ["text", "vector_score"],
                     hybrid_fields="*",
                     k: int = 5,
                     print_results: bool = False,
                     ):
        # Utworzenie wektora osadzenia na podstawie pytania użytkownika
        embedded_query = openai.Embedding.create(input=user_query,
                                                 model="text-embedding-ada-002",
                                                 )["data"][0]['embedding']
        # Przygotowanie zapytania
        base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
        query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict = {"vector": np.array(
            embedded_query).astype(dtype=np.float32).tobytes()}
        # Wyszukiwanie wektorowe
        results = self.redis_client.ft(index_name).search(query, params_dict)
        if print_results:
            for i, doc in enumerate(results.docs):
                score = 1 - float(doc.vector_score)
                print(f"{i}. {doc.text} (Score: {round(score ,3) })")
        return [doc['text'] for doc in results.docs]
