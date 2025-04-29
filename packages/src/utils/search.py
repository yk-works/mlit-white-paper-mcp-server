import torch
from lindera_py import Segmenter, Tokenizer, load_dictionary
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer
from utils.strings import ja_tokens


class Search:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.v_tokenizer = AutoTokenizer.from_pretrained(
            "pfnet/plamo-embedding-1b",
            trust_remote_code=True,
        )
        v_model = AutoModel.from_pretrained(
            "pfnet/plamo-embedding-1b",
            trust_remote_code=True,
        )
        self.v_model = v_model.to(device)

        dictionary = load_dictionary("ipadic")
        segmenter = Segmenter("normal", dictionary)
        self.tokenizer = Tokenizer(segmenter)
        self.r_model = CrossEncoder(
            "hotchpotch/japanese-bge-reranker-v2-m3-v1",
            max_length=512,
            device=device,
        )

    def fts_search(self, conn, query):
        q_tokens = ja_tokens(self.tokenizer, query)
        rows = conn.sql(f"""
            SELECT id, fts_main_mlit_doc.match_bm25(id, '{q_tokens}') AS score, content
            FROM mlit_doc
            WHERE score IS NOT NULL
            ORDER BY score DESC
        """).fetchall()

        return rows

    def vss_search(self, conn, query):
        with torch.inference_mode():
            query_embedding = self.v_model.encode_query(query, self.v_tokenizer)
            rows = conn.sql(
                """
                SELECT id, array_cosine_distance(content_v, ?::FLOAT[2048]) as distance, content
                FROM mlit_doc
                ORDER BY distance ASC
                """,
                params=[query_embedding.cpu().squeeze().numpy().tolist()],
            ).fetchall()

            return rows

    def reranking(self, query, vss_rows, fts_rows):
        # 凄く雑なマージ、あとから content から id をとれるようにしてる
        passages = {}
        for row in vss_rows:
            id, _, content = row
            passages[content] = id
        for row in fts_rows:
            id, _, content = row
            passages[content] = id

        contents = list(passages.keys())
        # Reranker
        scores = self.r_model.predict([(query, content) for content in contents])
        # スコア高い順にソートするタイミングで id と content を score に紐づける
        return sorted(
            [
                (passages[content], score, content)
                for content, score in zip(contents, scores)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def hybrid_search(self, conn, query, limit=10):
        fts_rows = sorted(
            self.fts_search(conn, query),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]
        vss_rows = sorted(
            self.vss_search(conn, query),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]
        return self.reranking(query, vss_rows, fts_rows)
