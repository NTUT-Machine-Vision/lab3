import uuid
from typing import List, Dict, Tuple, Optional, Any
from constants import NO_RES_ID, NO_RES_MSG
import numpy as np
import sqlite3
import json


class PaperStore:
    def __init__(self, db_path="paper_store.db"):
        self.db_path = db_path
        self._initialize_db()

    def _execute_query(
        self,
        query: str,
        params: tuple = (),
        fetch_one=False,
        fetch_all=False,
        commit=False,
    ):
        """Helper function to execute SQLite queries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = None
        if commit:
            conn.commit()
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        conn.close()
        return result

    def _initialize_db(self):
        """Initializes the database and creates the papers table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            desc TEXT,
            imgs TEXT,
            keywords TEXT 
        );
        """
        self._execute_query(query, commit=True)
        index_query_title = "CREATE INDEX IF NOT EXISTS idx_title ON papers (title);"
        self._execute_query(index_query_title, commit=True)

    def _serialize_imgs(self, imgs: List[Any]) -> str:
        """Serializes a list of images (URLs or numpy arrays) to a JSON string."""
        serialized_imgs = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                serialized_imgs.append({"type": "numpy_array", "data": img.tolist()})
            elif isinstance(img, str):
                serialized_imgs.append({"type": "url", "data": img})
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        return json.dumps(serialized_imgs)

    def _deserialize_imgs(self, imgs_json_str: Optional[str]) -> List[Any]:
        """Deserializes a JSON string back to a list of images."""
        if not imgs_json_str:
            return []
        imgs_data = json.loads(imgs_json_str)
        deserialized_imgs = []
        for item in imgs_data:
            if item["type"] == "numpy_array":
                deserialized_imgs.append(np.array(item["data"], dtype=np.uint8))
            elif item["type"] == "url":
                deserialized_imgs.append(item["data"])
            else:
                raise ValueError(f"Unsupported image type in JSON: {item['type']}")
        return deserialized_imgs

    def _serialize_keywords(self, keywords: List[str]) -> str:
        """Serializes a list of keywords to a JSON string."""
        return json.dumps(keywords)

    def _deserialize_keywords(self, keywords_json_str: Optional[str]) -> List[str]:
        """Deserializes a JSON string back to a list of keywords."""
        if not keywords_json_str:
            return []
        try:
            keywords = json.loads(keywords_json_str)
            if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                return keywords
            return []  # Return empty list if not a list of strings
        except json.JSONDecodeError:
            return []  # Return empty list if JSON is invalid

    def find_paper_id_by_title(self, title: str) -> Optional[str]:
        """Finds a paper ID by its exact title."""
        query = "SELECT id FROM papers WHERE title = ?"
        result = self._execute_query(query, (title,), fetch_one=True)
        return result[0] if result else None

    def add_paper(
        self,
        title: str,
        desc: str,
        imgs: List[Any],
        keywords: List[str],
    ) -> str:
        """Adds a new paper to the database with keywords."""
        paper_id = str(uuid.uuid4())
        query = "INSERT INTO papers (id, title, desc, imgs, keywords) VALUES (?, ?, ?, ?, ?)"
        params = (
            paper_id,
            title,
            desc,
            self._serialize_imgs(imgs),
            self._serialize_keywords(keywords),
        )
        self._execute_query(query, params, commit=True)
        return paper_id

    def get_paper_choices(self, ignore_no_res: bool = False) -> List[Tuple[str, str]]:
        """Returns a list of (title, id) tuples for paper selection, sorted by title."""
        query = "SELECT title, id FROM papers ORDER BY title ASC"
        results = self._execute_query(query, fetch_all=True)
        if not results:
            return [] if ignore_no_res else [(NO_RES_MSG, NO_RES_ID)]
        return [(row[0], row[1]) for row in results]

    def search_papers(self, keyword: str) -> List[Tuple[str, str]]:
        """Searches papers by keyword in title, description, or as an exact match in the keywords list."""
        keyword_lc = keyword.lower().strip()
        if not keyword_lc:
            return self.get_paper_choices()

        like_pattern_substring = f"%{keyword_lc}%"
        like_pattern_exact_keyword = f'%"{keyword_lc}"%'

        query = """
        SELECT title, id FROM papers 
        WHERE LOWER(title) LIKE ? 
           OR LOWER(desc) LIKE ?
           OR (keywords IS NOT NULL AND LOWER(keywords) LIKE ?)
        ORDER BY title ASC
        """
        params = (
            like_pattern_substring,
            like_pattern_substring,
            like_pattern_exact_keyword,
        )
        results = self._execute_query(query, params, fetch_all=True)

        if not results:
            return [(NO_RES_MSG, NO_RES_ID)]
        return [(row[0], row[1]) for row in results]

    def get_paper_details_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full details for a given paper ID, including keywords."""
        query = "SELECT id, title, desc, imgs, keywords FROM papers WHERE id = ?"
        result = self._execute_query(query, (paper_id,), fetch_one=True)
        if result:
            return {
                "id": result[0],
                "title": result[1],
                "desc": result[2],
                "imgs": self._deserialize_imgs(result[3]),
                "keywords": self._deserialize_keywords(result[4]),
            }
        return None

    def __del__(self):
        pass
