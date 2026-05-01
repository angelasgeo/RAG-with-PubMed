import os
import json
import time
import random
import re
import argparse
from Bio import Entrez
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = ""
MODEL_NAME = "gemini-3-flash-preview"

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

Entrez.email = "researcher@example.com"

def extract_keywords(question):
    prompt = f"Extract 2-4 primary medical keywords from this question to use as a PubMed search query. Return ONLY the keywords separated by spaces, no punctuation. Question: {question}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        clean = re.sub(r'(?i)^(is|are|does|do|what|how|why|can|could|would|should)\s+', '', question)
        return re.sub(r'[^\w\s]', '', clean)

def fetch_pubmed_abstracts(query, k=20):
    try:
        h = Entrez.esearch(db="pubmed", term=query, retmax=k)
        r = Entrez.read(h)
        h.close()
        pmids = r.get("IdList", [])
        if not pmids: return []

        h = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
        recs = Entrez.read(h)
        h.close()
        
        docs = []
        for article in recs.get("PubmedArticle", []):
            art = article["MedlineCitation"]["Article"]
            pmid = str(article["MedlineCitation"]["PMID"])
            title = str(art.get("ArticleTitle", ""))
            abstract_list = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(x) for x in abstract_list)
            if abstract:
                docs.append({"pmid": pmid, "title": title, "abstract": abstract})
        return docs
    except Exception as e:
        print(f"  [!] PubMed API Error: {e}")
        return []

def clean_json(text):
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()

def extract_triples_with_retry(text, max_retries=3):
    prompt = f"""
    You are a medical knowledge graph constructor. Extract clinical entities and their relationships.
    Return strictly a JSON object with a key "triples" containing a list of objects with "subject", "predicate", and "object".
    Text: {text}
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            data = json.loads(clean_json(content))
            return data.get("triples", [])
        except Exception as e:
            sleep_time = (attempt + 1) * 2
            print(f"  [!] Extraction error (Attempt {attempt+1}/{max_retries}): {e}. Sleeping {sleep_time}s...")
            time.sleep(sleep_time)
    return []

def ingest_to_neo4j(triples, pmid, evidence_map=None):
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session() as session:
            for t in triples:
                if not all(k in t for k in ("subject", "predicate", "object")): continue
                subj = str(t["subject"]).strip().lower()
                obj = str(t["object"]).strip().lower()
                pred = str(t["predicate"]).strip().upper().replace(" ", "_")
                pred = "".join(c for c in pred if c.isalnum() or c == "_")
                if not pred: pred = "RELATED_TO"
                
                # Get specific evidence for this triple if available, otherwise use a generic snippet
                evidence = t.get("evidence", "")
                
                cypher = f"""
                MERGE (s:Entity {{name: $subj}}) 
                MERGE (o:Entity {{name: $obj}}) 
                MERGE (s)-[r:{pred}]->(o)
                SET r.pmid = $pmid, r.evidence = $evidence
                """
                try:
                    session.run(cypher, subj=subj, obj=obj, pmid=pmid, evidence=evidence)
                except Exception:
                    pass 

def build_graph(resume_index=0):
    with open("ori_pqal.json", "r") as f:
        questions_data = json.load(f)
    with open("test_ids_250.json", "r") as f:
        test_ids = json.load(f)
    
    print(f"Starting FAST pipeline for {len(test_ids)} questions using local data...")
    if resume_index > 0:
        print(f"Resuming from index {resume_index}...")
        test_ids = test_ids[resume_index:]
        
    total_ingested = 0
    
    for qid in tqdm(test_ids):
        if qid not in questions_data: continue
        
        sections = questions_data[qid].get("CONTEXTS", [])
        text = " ".join(sections)
        
        if not text:
            continue
            
        triples = extract_triples_with_retry(text)
        
        if triples:
            ingest_to_neo4j(triples, pmid=qid)
            total_ingested += len(triples)
        
        time.sleep(0.1)
            
    print(f"\n=== GRAPH BUILD COMPLETE ===")
    print(f"Total relationships ingested into Neo4j: {total_ingested}")

def extract_triples_enhanced(text, pmid, max_retries=3):
    """
    Enhanced extraction with strict schema and evidence tracking.
    """
    allowed_predicates = ["TREATS", "CAUSES", "ASSOCIATED_WITH", "DIAGNOSES", "PREVENTS", "INHIBITS", "STIMULATES", "LOCATION_OF", "PART_OF"]
    prompt = f"""
    You are a medical knowledge graph expert. Extract clinical entities and their relationships from the text below.
    
    STRICT RULES:
    1. Use ONLY these predicates: {allowed_predicates}.
    2. Extract the "evidence" (the exact sentence or phrase where the triple was found).
    3. Subject and Object should be concise medical terms.
    
    Return strictly a JSON object with a key "triples" containing a list of objects:
    {{"subject": "...", "predicate": "...", "object": "...", "evidence": "..."}}
    
    Text: {text}
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(clean_json(content))
            triples = data.get("triples", [])
            # Filter and normalize
            valid_triples = []
            for t in triples:
                p = str(t.get("predicate", "")).upper().replace(" ", "_")
                if p in allowed_predicates:
                    valid_triples.append({
                        "subject": t.get("subject", "").lower().strip(),
                        "predicate": p,
                        "object": t.get("object", "").lower().strip(),
                        "evidence": t.get("evidence", ""),
                        "pmid": pmid
                    })
            return valid_triples
        except Exception as e:
            time.sleep(2)
    return []

def ingest_enhanced_to_neo4j(triples):
    """
    Ingests triples with evidence and source metadata.
    """
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session() as session:
            for t in triples:
                # Store evidence and pmid as relationship properties
                cypher = f"""
                MERGE (s:Entity {{name: $subj}})
                MERGE (o:Entity {{name: $obj}})
                CREATE (s)-[r:{t['predicate']}]->(o)
                SET r.evidence = $evidence, r.pmid = $pmid
                """
                try:
                    session.run(cypher, subj=t['subject'], obj=t['object'], 
                                evidence=t['evidence'], pmid=t['pmid'])
                except Exception as e:
                    print(f"Ingestion error: {e}")

def build_enhanced_graph(resume_index=0):
    # Similar to build_graph but uses the enhanced functions
    with open("ori_pqal.json", "r") as f:
        questions_data = json.load(f)
    with open("test_ids_250.json", "r") as f:
        test_ids = json.load(f)

    print(f"Building ENHANCED graph for {len(test_ids)} questions using local data...")
    if resume_index > 0:
        print(f"Resuming from index {resume_index}...")
        test_ids = test_ids[resume_index:]

    total_triples = 0
    for qid in tqdm(test_ids):
        if qid not in questions_data: continue
        
        # Combine local context sections
        sections = questions_data[qid].get("CONTEXTS", [])
        text = " ".join(sections)
        
        if not text: continue
        
        # Use qid as pmid for reference in the local context
        triples = extract_triples_enhanced(text, qid)
        if triples:
            ingest_enhanced_to_neo4j(triples)
            total_triples += len(triples)
            
        time.sleep(0.2)
        
    print(f"Enhanced Build Complete: {total_triples} relationships with evidence.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Knowledge Graph Builder")
    parser.add_argument("--enhanced", action="store_true", help="Build enhanced graph with evidence tracking")
    parser.add_argument("--resume", "-r", type=int, default=0, help="Index to resume from (0-based)")
    
    args = parser.parse_args()
    
    if args.enhanced:
        build_enhanced_graph(resume_index=args.resume)
    else:
        build_graph(resume_index=args.resume)