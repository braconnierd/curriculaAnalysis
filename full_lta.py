# ------------------------------------------------------------------------------
# Import and Library Management
# ------------------------------------------------------------------------------
import os
from sentence_transformers import SentenceTransformer,util
import numpy as np
import re
from pathlib import Path
import torch
import json
from collections import defaultdict

# -----------------------------------
# Input and Output Folders
# -----------------------------------

input_folder = Path("")     # where input TXT files of course content are
topics_folder = Path("")    # where TXT files of topics lists are
evidence_folder = Path("")  # where the output with the individual scores folder
evidence_folder.mkdir(parents=True, exist_ok=True)
SchoolName = "" # Name of School (mainly used for file naming purposes)

# ---------------------------
# Configuration / thresholds
# ---------------------------
#Embedding model
model = SentenceTransformer(
    "google/embeddinggemma-300m",
    device="mps"
)

batch_size = 16
docs_per_chunk = 5
# ---------------------------
# Helper Libraries
# ---------------------------

# Filler terms to filter out of the topic lists
generic_terms = {
    "analysis","approach","aspect","background","behavior","case",
    "characteristic","concept","effect","environment","evaluation",
    "example","factor","feature","framework","impact",
    "importance","influence","introduction","issue","measurement",
    "method","overview","process","property","result","role",
    "study","technique","theory","type","use", "is", "of", "and", 
    "or", "are", "for", "with", "the", "be", "to", "a", "in", 
    "that", "have", "it", "not", "on", "with","as", "this", "but", 
    "by", "from", "or", "an", "all", "which", "about", "there", 
    "their", "out", "of", "if", "who", "get", "when", "like", "know",
    "than", "then", "most", "these", "any", "even", "how", "usage",
    "during", "formation", "vs"
}

# ---------------------------
# Topic List Handling
# ---------------------------

def parse_all_topic_lists(topics_root):
    """Check through all of the topics lists within topics_root, and parses every .txt topic list.

    Returns:
        dict[str, dict]: A dictionary with the following structure:
            {
                "Building Block A": {
                    "Topic List 1": nested dict,
                    "Topic List 2": nested dict
                },
                "Building Block B": {
                    "Topic List 1": nested dict,
                    "Topic List 2": nested dict
                }
            }

    """


    all_topics = {}

    for discipline in os.listdir(topics_root):
        discipline_path = os.path.join(topics_root, discipline)

        if not os.path.isdir(discipline_path):
            continue

        all_topics[discipline] = {}

        for fname in os.listdir(discipline_path):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(discipline_path, fname)
            topic_name = os.path.splitext(fname)[0]

            parsed = parse_topic_list(fpath)
            all_topics[discipline][topic_name] = parsed

    return all_topics

def parse_topic_list(path):
    """Parses an individual hierarchical topic list with optional weights and nested subtopics.

    Returns:
        dict[str, dict]: A nested dictionary with the following structure:
            {
                "main_topic": {
                    "weight": float | None,
                    "subtopics": dict
                }
            }

    """
    topic_hierarchy = {}
    stack = []  # keeps track of current path
    indent_stack = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue

            # Compute indentation level via number of leading spaces
            indent = len(line) - len(line.lstrip())

            # Clean the text - remove bullets, numbering
            clean = re.sub(r"^\d+[\).\s]+", "", raw.lstrip())
            clean = re.sub(r"^[\-\*\•\–]+\s*", "", clean).strip()
            if not clean:
                continue

            # Extract weight if present (e.g., (60%) or (%40))
            weight_match = re.search(r"\((\d+)%\)", clean)
            weight = float(weight_match.group(1)) / 100 if weight_match else None
            if weight_match:
                clean = re.sub(r"\(\d+%\)", "", clean).strip()

            # Determine where this topic fits in hierarchy
            while indent_stack and indent <= indent_stack[-1]:
                stack.pop()
                indent_stack.pop()

            if not stack:
                # top-level topic
                topic_hierarchy[clean] = {"weight": weight, "subtopics": {}}
                stack.append(clean)
                indent_stack.append(indent)
            else:
                # nested subtopic
                parent = topic_hierarchy
                for t in stack:
                    parent = parent[t]["subtopics"]
                parent[clean] = {"weight": weight, "subtopics": {}}
                stack.append(clean)
                indent_stack.append(indent)

    return topic_hierarchy

def prepare_all_topics(all_topics, model, batch_size=16):
    """
    Flatten and embed each topic list independently.

    Returns:
        dict[str, dict[str, list[dict[str, object]]]]: A nested dictionary with
        the following structure:
            {
                "Building Block": {
                    "Topic List": embedded_flat_topics
                }
            }

        Each ``embedded_flat_topics`` value is a list of flattened checklist
        item dictionaries with corresponding embedding vectors.
    """
    prepared = {}

    for discipline, lists in all_topics.items():
        prepared[discipline] = {}

        for topic_name, hierarchy in lists.items():
            flat = flatten_topics(hierarchy, topic_name)
            if not flat:
                print(f"Skipping empty topic list: {discipline} → {topic_name}")
                continue
            embedded = embed_flat_topics(flat, model, batch_size=batch_size)
            prepared[discipline][topic_name] = embedded

    return prepared

def flatten_topics(topic_tree, topic_file):
    """Flattens a hierarchical topic tree into atomic, context-aware checklist items.

    Returns:
        list[dict[str, object]]: A list of dictionaries, where each dictionary
        represents an individual checklist item with the following structure:
            {
                "topic_name": str,
                "topic_file": str,
                "subtopic": str,
                "subtopic_weight": float | None,
                "check_item": str,
                "context": list[str],
                "full_query": str
            }

    """

    topic_name = os.path.splitext(os.path.basename(topic_file))[0]

    flat = []

    def recurse(node, subtopic_name, subtopic_weight, context_chain):
        for label, info in node.items():

            # Split multi-part lines into atomic checklist items
            atoms = extract_atomic_topics(label)

            for atom in atoms:
                context_text = " and ".join([topic_name] + context_chain)
                flat.append({
                    "topic_name": topic_name,
                    "topic_file": topic_file,

                    "subtopic": subtopic_name,
                    "subtopic_weight": subtopic_weight,

                    "check_item": atom,
                    "context": context_chain.copy(),
                    "full_query" : (
                        f"Content about {atom} "
                        f"as it relates to {context_text}"
                    )
                })

            if info["subtopics"]:
                recurse(
                    info["subtopics"],
                    subtopic_name,
                    subtopic_weight,
                    context_chain + [label]
                )

    for subtopic, data in topic_tree.items():
        recurse(
            data["subtopics"],
            subtopic_name=subtopic,
            subtopic_weight=data["weight"],
            context_chain=[subtopic]
        )

    return flat

def extract_atomic_topics(text):
    """Extracts atomic topic concepts from a complex topic string.

    The function decomposes topic strings into individual concepts by handling
    common formatting patterns, including:
    - Colon-separated topic headings.
    - Comma-separated lists.
    - Examples within parenthesis.
    - Lists joined by "and".
    - Dash-separated items.
    - Leading numbering or bullet symbols.

    Returns:
        list[str]: A list of atomic topic strings extracted from the
            input text.
    """

    atomic = []

    # 1. Split by colon: "Key processes: FDM, SLM" → ["Key processes", "FDM, SLM"]
    if ":" in text:
        left, right = text.split(":", 1)
        atomic.append(left.strip())
        text = right.strip()

    # 2. Remove leading numbering or symbols (just in case)
    text = re.sub(r"^\d+[\).\s]+", "", text)
    text = re.sub(r"^[\-\*\•\–]+\s*", "", text)

    # 3. Extract parenthetical examples: "Polymers (PLA, ABS)" → ["Polymers", "PLA", "ABS"]
    paren = re.findall(r"\((.*?)\)", text)
    for p in paren:
        items = [i.strip() for i in re.split(r",", p) if i.strip()]
        atomic.extend(items)

    # Remove the parentheses content from the main text
    text = re.sub(r"\(.*?\)", "", text).strip()

    # 4. Split by commas: "FDM, SLM, EBM" → ["FDM", "SLM", "EBM"]
    if "," in text:
        items = [i.strip() for i in text.split(",") if i.strip()]
        atomic.extend(items)
        return list(set(atomic))

    # 5. Split on "and": "geometry constraints and overhangs"
    if " and " in text:
        parts = [p.strip() for p in text.split(" and ") if p.strip()]
        atomic.extend(parts)
        return list(set(atomic))

    # 6. Split on hyphens used as lists: "operating principles - energy mechanisms"
    if " - " in text:
        parts = [p.strip() for p in text.split(" - ") if p.strip()]
        atomic.extend(parts)
        return list(set(atomic))

    # Default: treat the cleaned text itself as a single atomic concept
    if text:
        atomic.append(text.strip())

    return list(set(atomic))

def embed_flat_topics(flat_topics, model, batch_size):
    """Generates embeddings for flattened topic checklist items.

    Returns:
        list[dict[str,object]]: An input list with each checklist item's embedding vectors (for the individual atom and full context query) as NumPy arrays.
    """
    atoms = [t["check_item"] for t in flat_topics]
    queries = [t["full_query"] for t in flat_topics]

    atom_embeddings = model.encode(
        atoms,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    for i, t in enumerate(flat_topics):
        t["atom_embedding"] = atom_embeddings[i]
        t["query_embedding"] = query_embeddings[i]

    return flat_topics


# ------------------------------------------------------------------------------
# Embedding the Course Content and Helper Functions
# ------------------------------------------------------------------------------

def stream_corpus_sentences(root_dir, batch_size, docs_per_chunk=5):
    """Processes the txt documents within the course content directory in chunks. Splits each text document and encodes individual sentences with additional filtering to remove duplicates and near-duplicates.

    Yields:
        dict[str, dict[str, list | ndarray]]: A dictionary mapping each
        document path to its processed content, with the following structure:
            {
                "document_path": {
                    "sentences": list[str],
                    "embeddings": ndarray
                }
            }

    """
    corpus_chunk = {}

    for course in os.listdir(root_dir):
        course_dir = os.path.join(root_dir, course)
        if not os.path.isdir(course_dir):
            continue

        for fname in os.listdir(course_dir):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(course_dir, fname)

            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

            
            raw_sentences = split_into_sentences(text)

            # Filtering out exact duplicates
            unique_sentences = exact_dedup(raw_sentences)

            if not unique_sentences:
                continue

            # Filter out excessively short and excessively long sentences, and limit the number of sentences to embedded
            unique_sentences = [
                s for s in unique_sentences
                if 40 < len(s) < 800
            ]
            MAX_PRE_EMBED = 400
            unique_sentences = unique_sentences[:MAX_PRE_EMBED]

            if not unique_sentences:
                continue

            embeddings = model.encode(
                unique_sentences,
                batch_size, 
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # filter out near-duplicates according to the embeddings
            cleaned_sentences, cleaned_embeddings = remove_near_duplicates_from_embeddings(
                unique_sentences,
                embeddings,
                threshold=0.90
            )

            if not cleaned_sentences:
                continue

            MAX_SENTENCES_PER_DOC = 250
            cleaned_sentences = cleaned_sentences[:MAX_SENTENCES_PER_DOC]
            cleaned_embeddings = cleaned_embeddings[:MAX_SENTENCES_PER_DOC]

            corpus_chunk[fpath] = {
                "sentences": cleaned_sentences,
                "embeddings": cleaned_embeddings
            }

            # Return chunk and empty cache
            if len(corpus_chunk) >= docs_per_chunk:
                yield corpus_chunk
                corpus_chunk = {}
                torch.mps.empty_cache()

    if corpus_chunk:
        yield corpus_chunk

def split_into_sentences(text):
    """
    Splits sentences according to punctuation marks

    Returns:
        list[str]: A list of strings with each string being an individual sentence
    """
    
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def exact_dedup(sentences):
    """
    Exact deduplication of sentences using normalized keys, while keeping the first occurrence.
    
    Returns:
        list[str]: list of all of the unique sentences.
    """
    seen = set()
    unique = []

    for s in sentences:
        s = s.lower()
        s = re.sub(r"\b\d+(\.\d+)*\b", "", s)       #remove numbers
        s = re.sub(r"[^\w\s]", "", s)               #remove punctuation
        s = re.sub(r"\s+", " ", s).strip()          #normalize spaces

        if not s:
            continue

        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique

def remove_near_duplicates_from_embeddings(sentences, embeddings, threshold=0.93):
    """Remove all of the near duplicate sentences using cosine similarity of normalized embeddings.

    Returns:
        tuple[list[str], ndarray]: A tuple containing:
            - A list of sentences with all near duplicates removed.
            - A two-dimensional array of embedding vectors corresponding to
              the retained sentences.

    """
    kept_indices = []
    kept_matrix = []

    for i, emb in enumerate(embeddings):

        if not kept_matrix:
            kept_indices.append(i)
            kept_matrix.append(emb)
            continue

        kept_array = np.vstack(kept_matrix)

        sims = kept_array @ emb

        if np.max(sims) < threshold:
            kept_indices.append(i)
            kept_matrix.append(emb)

    kept_sentences = [sentences[i] for i in kept_indices]
    kept_embeddings = np.vstack([embeddings[i] for i in kept_indices])

    return kept_sentences, kept_embeddings


# ------------------------------------------------------------------------------
# LTA-related Functions
# ------------------------------------------------------------------------------

def run_lta(
    flat_topics,
    corpus,
    atom_threshold=0.55,
    query_threshold=0.45,
    context_threshold=0.36
):
    """Matches flattened topic checklist items against course content, and generates a dictionary with the relevant metadata.
    
    Returns:
        list[dict[str, object]]: A list of topic dictionaries with
        retrieval results, including:
            {
                ...
                "hits": list[dict],
                "total_hits": int,
                "missed": bool,
                "best_evidence": dict,
                "atom_heads": list[str]
            }
    """
    results = []

    atom_matrix = torch.stack([
        torch.as_tensor(t["atom_embedding"], dtype=torch.float32)
        for t in flat_topics
    ])

    query_matrix = torch.stack([
        torch.as_tensor(t["query_embedding"], dtype=torch.float32)
        for t in flat_topics
    ])

    topic_meta = [
        {
            "atom_heads": extract_head_terms(t["check_item"]),
            "atom_kind": atom_type(t["check_item"])
        }
        for t in flat_topics
    ]

    for idx, t in enumerate(flat_topics):
        results.append({
            **t,
            "hits": [],
            "total_hits": 0,
            "missed": True,
            "best_evidence": {
                "best_atom_match": {
                    "atom_score": -1,
                    "query_score": None,
                    "document": None,
                    "sentence": None,
                },
                "best_query_match": {
                    "score": -1,
                    "atom_score": None,
                    "document": None,
                    "sentence": None,
                },
                "best_lexical_match": {
                    "query_score": -1,
                    "atom_score": None,
                    "document": None,
                    "sentence": None,
                },
            },
            "atom_heads": topic_meta[idx]["atom_heads"]
        })

    for doc_path, data in corpus.items():

        sent_emb = torch.as_tensor(data["embeddings"], dtype=torch.float32)
        sentences = data["sentences"]

        atom_sims = torch.matmul(sent_emb, atom_matrix.T)
        query_sims = torch.matmul(sent_emb, query_matrix.T)

        for t_idx, topic in enumerate(results):

            atom_heads = topic_meta[t_idx]["atom_heads"]
            atom_kind = topic_meta[t_idx]["atom_kind"]
            be = topic["best_evidence"]

            a_sims = atom_sims[:, t_idx]
            q_sims = query_sims[:, t_idx]

            semantic_mask = (
                (a_sims >= atom_threshold) &
                (q_sims >= query_threshold)
            )

            if atom_kind == "named_entity":
                semantic_mask[:] = False

            doc_hits = []

            # ---------------- Semantic hits ----------------
            for i in torch.where(semantic_mask)[0]:
                i = int(i)
                doc_hits.append({
                    "sentence": sentences[i],
                    "atom_similarity": float(a_sims[i]),
                    "query_similarity": float(q_sims[i]),
                    "pathway": "semantic",
                    "lexical_heads": None
                })

            # ---------------- Lexical + best evidence ----------------
            for i, sentence in enumerate(sentences):

                a_score = float(a_sims[i])
                q_score = float(q_sims[i])

                # only track best evidence while still missed
                if topic["missed"]:

                    # Best atom match
                    if a_score > be["best_atom_match"]["atom_score"]:
                        be["best_atom_match"].update({
                            "atom_score": a_score,
                            "query_score": q_score,
                            "document": doc_path,
                            "sentence": sentence,
                        })

                    # Best query match (conditioned on atom threshold)
                    if (
                        a_score >= atom_threshold
                        and q_score > be["best_query_match"]["score"]
                    ):
                        be["best_query_match"].update({
                            "score": q_score,
                            "atom_score": a_score,
                            "document": doc_path,
                            "sentence": sentence,
                        })

                lex_hit = lexical_match(atom_heads, sentence)

                # Best lexical match
                if (
                    topic["missed"]
                    and lex_hit
                    and q_score > be["best_lexical_match"]["query_score"]
                ):
                    be["best_lexical_match"].update({
                        "query_score": q_score,
                        "atom_score": a_score,
                        "document": doc_path,
                        "sentence": sentence,
                    })

                # Final retrieval hit
                if lex_hit and q_score >= context_threshold:
                    doc_hits.append({
                        "sentence": sentence,
                        "atom_similarity": a_score,
                        "query_similarity": q_score,
                        "pathway": "lexical",
                        "lexical_heads": atom_heads
                    })

            # ---------------- Finalize list entries ----------------
            if doc_hits:
                topic["hits"].append({
                    "document": doc_path,
                    "hit_count": len(doc_hits),
                    "sentences": doc_hits
                })
                topic["total_hits"] += len(doc_hits)
                topic["missed"] = False

    return results

def extract_head_terms(atom, max_terms=4):
    """Extracts the most informative lexical terms from a topic.

    Returns:
        list[str]: A list of the most informative content terms extracted from
            the atomic topic. Returns an empty list if no content terms are
            found.

    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", atom.lower())
    content = [t for t in tokens if t not in generic_terms]
    return content[-max_terms:] if content else []

def atom_type(atom):
    """Determine what kind of atom it is, which determines what kind of matching is best.

    Returns:
        [str]: A string descibing the kind of atom it is.
    """
    atom_clean = atom.strip()

    # Acronyms / materials / standards
    if atom_clean.isupper() and len(atom_clean.split()) <= 2:
        return "named_entity"

    # Chemical-looking tokens (PLA, PETG, ABS, Ti64, etc.)
    if len(atom_clean.split()) == 1 and any(c.isdigit() for c in atom_clean):
        return "named_entity"

    # Everything else is a conceptual topic
    return "concept"

def lexical_match(atom_heads: list[str], sentence: str) -> bool:
    """Checks for exact matches for the atom (while accounting for plurals and variants on the word)

    Returns:
        [bool]: True if the sentence contains an exact match for the atom, False if there is no match.
    """

    if not atom_heads:
        return False
    
    sent = sentence.lower()

    # ---- Case 1: Acronym ----
    if len(atom_heads) == 1 and len(atom_heads[0]) <= 5:
        base = atom_heads[0].lower()
        variants = {
            base,
            f"({base})",
            f"{base}s"
        }

        for v in variants:
            pattern = rf"\b{re.escape(v)}\b"
            if re.search(pattern, sent):
                return True

    # ---- Case 2: Multi-word compound ----
    if len(atom_heads) > 1:
        phrase = " ".join(atom_heads).lower()
        pattern = rf"\b{re.escape(phrase)}s?\b"
        return re.search(pattern, sent) is not None

    # ---- Case 3: Single conceptual word ----
    head = atom_heads[0].lower()

    variants = {head}

    if head.endswith("ing"):
        root = head[:-3]
        variants.update({root, root + "ed", root + "s"})

    for v in variants:
        pattern = rf"\b{re.escape(v)}\b"
        if re.search(pattern, sent):
            return True

    return False


# ------------------------------------------------------------------------------
# Evidence File Generation + Result Management
# ------------------------------------------------------------------------------

def merge_lta_results(partial_results_list):
    """
    Merge run_lta outputs into one structure.

    Returns:
        dict[str, dict[str, list[dict]]]: A merged hierarchical structure:
            {
                "discipline": {
                    "topic_name": [merged_entry_dict, ...]
                }
            }

        Each merged entry aggregates: hits (list), total_hits (int), missed (bool), original metadata fields from run_lta()
    """

    merged = {}

    for partial in partial_results_list:
        for discipline, topic_lists in partial.items():

            if discipline not in merged:
                merged[discipline] = {}

            for topic_name, entries in topic_lists.items():

                if topic_name not in merged[discipline]:
                    merged[discipline][topic_name] = {}

                for entry in entries:
                    key = (
                        entry["subtopic"],
                        entry["check_item"]
                    )

                    if key not in merged[discipline][topic_name]:
                        merged[discipline][topic_name][key] = {
                            **entry,
                            "hits": [],
                            "total_hits": 0,
                            "missed": True
                        }

                    merged_entry = merged[discipline][topic_name][key]

                    # Merge hits
                    merged_entry["hits"].extend(entry["hits"])
                    merged_entry["total_hits"] += entry["total_hits"]

                    if entry["total_hits"] > 0:
                        merged_entry["missed"] = False

    final = {}

    for discipline, topic_lists in merged.items():
        final[discipline] = {}

        for topic_name, entries in topic_lists.items():
            final[discipline][topic_name] = list(entries.values())

    return final

def write_evidence_txt(lta_results, evidence_root):
    """
    Writes hierarchical evidence files.

    Returns:
        Text files are located at Evidence/[Building Block]/[Topic Name] Evidence.txt
    """

    from collections import defaultdict
    import os

    def hyperlink(path):
        return f"file://{os.path.abspath(path)}"

    for discipline, topic_lists in lta_results.items():

        discipline_dir = os.path.join(evidence_root, discipline)
        os.makedirs(discipline_dir, exist_ok=True)

        for topic_name, entries in topic_lists.items():

            out_path = os.path.join(
                discipline_dir,
                f"{topic_name} Evidence.txt"
            )

            # -------- Group by subtopic --------
            subtopics = defaultdict(list)
            for e in entries:
                subtopics[e["subtopic"]].append(e)

            with open(out_path, "w", encoding="utf-8") as f:

                f.write(f"{topic_name} — Evidence Report\n")

                total_score = 0.0
                remaining_weight = 1.0
                unweighted_scores = []

                for subtopic, items in subtopics.items():
                    item_by_name = {i["check_item"].lower(): i for i in items}

                    for item in items:
                        if item["missed"]:
                            continue

                        # Walk upward through context - propagates hits if the child is covered, the parent is considered covered.
                        for parent in item.get("context", []):
                            parent_key = parent.lower()
                            if parent_key in item_by_name:
                                parent_item = item_by_name[parent_key]
                                if parent_item["missed"]:
                                    parent_item["missed"] = False
                                    parent_item["total_hits"] = 0
                                    parent_item.setdefault("hits", [])

                    weight = items[0]["subtopic_weight"]

                    covered = sum(not i["missed"] for i in items)
                    raw_score = covered / len(items)

                    if weight is not None:
                        total_score += weight * raw_score
                        remaining_weight -= weight
                    else:
                        unweighted_scores.append(raw_score)

                if unweighted_scores and remaining_weight > 0:
                    total_score += remaining_weight * (
                        sum(unweighted_scores) / len(unweighted_scores)
                    )

                f.write(f"Score: {total_score * 100:.2f}%\n")
                f.write("=" * 70 + "\n\n")

                # ================= SUBTOPICS =================
                for subtopic, items in subtopics.items():

                    weight = items[0]["subtopic_weight"]
                    weight_str = (
                        f"{int(weight * 100)}%" if weight is not None
                        else "Unweighted"
                    )

                    covered = sum(not i["missed"] for i in items)
                    raw_score = covered / len(items) * 100

                    f.write(
                        f"{subtopic} — Weight: {weight_str} — "
                        f"Raw Score: {raw_score:.1f}%\n"
                    )
                    f.write("=" * 70 + "\n")

                    for item in items:

                        status = "Covered" if not item["missed"] else "Missed"
                        f.write(f"{item['check_item']} — {status}")

                        if not item["missed"]:
                            f.write(f" — {item['total_hits']} hits\n")
                        else:
                            f.write("\n")

                        f.write(f"Atom heads: {item['atom_heads']}\n")

                        # -------- Covered --------
                        if not item["missed"]:
                            for d in item["hits"]:
                                fname = os.path.basename(d["document"])
                                link = hyperlink(d["document"])

                                f.write(f"    * {fname} — {d['hit_count']} hits\n")
                                f.write(f"        {link}\n")

                                for s in d["sentences"]:
                                    f.write(
                                        f"        ‣ \"{s['sentence']}\" "
                                        f"[{s['pathway']}] "
                                        f"(atom={s['atom_similarity']:.3f}, "
                                        f"query={s['query_similarity']:.3f})\n"
                                    )

                        # -------- Missed → Best Evidence --------
                        else:
                            be = item["best_evidence"]

                            for label, key in [
                                ("Best lexical match", "best_lexical_match"),
                                ("Best atom match", "best_atom_match"),
                                ("Best contextual match", "best_query_match")
                            ]:
                                match = be.get(key)
                                if match and match.get("document"):
                                    fname = os.path.basename(match["document"])
                                    f.write(f"    * {label}: {fname}\n")
                                    f.write(f"        {hyperlink(match['document'])}\n")
                                    f.write(
                                        f"        ‣ \"{match['sentence']}\" "
                                        f"(atom={match.get('atom_score', 0):.3f}, "
                                        f"query={match.get('query_score', match.get('score', 0)):.3f})\n"
                                    )

                        f.write("\n")

                    f.write("\n")

            print(f"[✓] Evidence written: {out_path}")

def export_scores_json(merged_results, output_path, school_name):
    """Generates a json with the results per topic list.

    Returns: 
        A text file located at Evidence/[SchoolName].json
    """

    output = {
        "school_name": school_name,
        "results": {}
    }

    for discipline, topic_lists in merged_results.items():
        output["results"][discipline] = {}

        for topic_name, entries in topic_lists.items():

            # --- replicate score logic ---
            subtopics = defaultdict(list)
            for e in entries:
                subtopics[e["subtopic"]].append(e)

            total_score = 0.0
            remaining_weight = 1.0
            unweighted_scores = []

            for subtopic, items in subtopics.items():

                weight = items[0]["subtopic_weight"]
                covered = sum(not i["missed"] for i in items)
                raw_score = covered / len(items)

                if weight is not None:
                    total_score += weight * raw_score
                    remaining_weight -= weight
                else:
                    unweighted_scores.append(raw_score)

            if unweighted_scores and remaining_weight > 0:
                total_score += remaining_weight * (
                    sum(unweighted_scores) / len(unweighted_scores)
                )

            output["results"][discipline][topic_name] = round(total_score * 100, 2)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[✓] JSON exported: {output_path}")


# ------------------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------------------

processed_docs = 0
total_docs = 0

#Count all the documents within the course content
for root, _, files in os.walk(input_folder):
    for fname in files:
        if fname.endswith(".txt"):
            total_docs += 1


#Prepare the topic lists
all_topic_lists = parse_all_topic_lists(topics_folder)
prepared_topics = prepare_all_topics(all_topic_lists, model, batch_size)

num_topic_lists = sum(
    len(lists) for lists in prepared_topics.values()
)

merged_results = {}

#Iterate through all of the course content, and check through all possible answers
for idx, corpus_chunk in enumerate(
    stream_corpus_sentences(input_folder,batch_size, docs_per_chunk)
):

    print(f"Processing corpus chunk {idx + 1}")

    partial_results = {}

    for discipline, topic_lists in prepared_topics.items():
        partial_results[discipline] = {}

        for topic_name, embedded_topics in topic_lists.items():
            result = run_lta(embedded_topics, corpus_chunk)
            partial_results[discipline][topic_name] = result

    merged_results = merge_lta_results(
        [merged_results, partial_results]
    )

    processed_docs += len(corpus_chunk)


    print(
        f"Processed {processed_docs}/{total_docs} docs"
    )

    torch.mps.empty_cache() #Clear cache to prevent overloading the memory

write_evidence_txt(merged_results, evidence_folder)
export_scores_json(
    merged_results,
    os.path.join(evidence_folder, f"{SchoolName}.json"),
    school_name=SchoolName
)