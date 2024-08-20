import json
import logging
import os
import subprocess
import traceback
import weaviate
import uuid
from pathlib import Path
from typing import Tuple, List

import numpy as np
import requests
import torch
from langchain_community.embeddings import (HuggingFaceBgeEmbeddings,
                                            HuggingFaceEmbeddings,
                                            OpenAIEmbeddings)
from langchain_community.vectorstores import Weaviate
from llmsherpa.readers import Document, LayoutPDFReader, Section
from numpy.linalg import norm
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx

from langchain.embeddings.base import Embeddings
from langchain.schema.vectorstore import VectorStore

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import TextSplitter


def get_embeddings(embeddings_model_name: str = "BAAI/llm-embedder", securellm=False) -> Embeddings:
    # set the device to cuda if the GPU is available otherwise use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Embeddings model name: {embeddings_model_name} using device: {device}")

    if embeddings_model_name == "mpnet-v2":
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": device}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        return hf

    elif embeddings_model_name == "openai":
        if securellm:
            embeddings = OpenAIEmbeddings(client=None)
        else:
            embeddings = OpenAIEmbeddings(
                client=None,
                openai_api_base="https://api.openai.com/v1",
                openai_api_type="open_ai",
            )

        return embeddings

    elif embeddings_model_name == "bert":
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)

    else:
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceBgeEmbeddings(
            model_name=embeddings_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)


def create_paperqa_vector_indexes(client: VectorStore, embeddings: Embeddings,
                                  dataset: str, follow_on_questions: bool = False,
                                  enable_cache: bool = False) -> Tuple[VectorStore, VectorStore]:
    # Create 2 classes (or DBs) in Weaviate
    # One for docs and one for chunks
    docs = Weaviate(
        client=client,
        index_name="D" + dataset + "docs",
        text_key="paperdoc",
        attributes=['dockey'],
        embedding=embeddings,
        by_text=False,
    )

    attributes = ['doc', 'name', 'is_table', 'ext_path', 'doc_vector_ids',
                  'doc_source', 'state_category', 'relevant_vectors', 'embed_text']
    paper_chunks_class_name = "D" + dataset + "chunks"

    try:
        schema = client.schema.get(paper_chunks_class_name)
        properties = schema.get("properties")
        schema_attributes = [p.get('name') for p in properties]

        if 'dockey' in schema_attributes:
            attributes.append('dockey')

    except Exception as e:
        logging.error(f"No schema found, use default attributes: error{e}")
        attributes.append('dockey')

    if follow_on_questions:
        attributes.append('follow_on_question')

    chunks = Weaviate(
        client=client,
        index_name=paper_chunks_class_name,
        text_key="paperchunks",
        embedding=embeddings,
        attributes=attributes,
        by_text=False,
    )

    cache = None
    if enable_cache:
        cache = Weaviate(
            client=client,
            index_name="D" + dataset + "cache",
            text_key="qnacache",
            embedding=embeddings,
            attributes=['doc', 'trace_id', 'state_category',
                        'designation_category', 'references', 'question'],
            by_text=False,
        )

    return docs, chunks, cache


'''
Weaviate URL from inside the cluster
--- "http://weaviate.d3x.svc.cluster.local/"
Weaviate URL from outside the cluster
--- NodePort - "http://<ip>:30716/api/vectordb/"
--- Loadbalancer - "http://<ip>:80/api/vectordb/"
'''


def get_vectordb_client() -> VectorStore:
    WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)
    DKUBEX_API_KEY = os.getenv("DKUBEX_API_KEY", "deadbeef")

    # Use Weaviate VectorDB
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"Authorization": DKUBEX_API_KEY},
    )

    return weaviate_client


def get_metrics(answer, vector_ids, qa_data, sim_model):
        hit = 0
        _mrr = 0
        sim_score = 0
        refans_embeddings = sim_model.encode(qa_data['answer'].lower().strip(), normalize_embeddings=True)
        genans_embeddings = sim_model.encode(str(answer).lower().strip(), normalize_embeddings = True)
        sim_score = np.dot(refans_embeddings, genans_embeddings)/(norm(refans_embeddings) * norm(genans_embeddings))
        matched_vector_id = list(set(qa_data["relevant_docs"]).intersection(set(vector_ids)))
        if matched_vector_id:
            matched_vector_id = matched_vector_id[0]
            hit = 1
            # total_hit += 1
            _mrr = 1/(vector_ids.index(matched_vector_id)+1)
            # total_mrr += _mrr
        else:
            hit = 0
            _mrr = 0

        return hit, _mrr, sim_score


def get_sim_model():
    return "BAAI/bge-large-en-v1.5"


def get_pdfinfo(infile):
    """Get Information about the pdf file"""
    cmd = "/usr/bin/pdfinfo"
    if not os.path.exists(cmd):
        raise RuntimeError("System command not found: %s" % cmd)

    if not os.path.exists(infile):
        raise RuntimeError("Provided input file not found: %s" % infile)

    def _extract(row):
        """Extracts the right hand value from a : delimited row"""
        return row.split(":", 1)[1].strip()

    output = {}

    labels = [
        "Title",
        "Author",
        "Creator",
        "Producer",
        "CreationDate",
        "ModDate",
        "Tagged",
        "Pages",
        "Encrypted",
        "Page size",
        "File size",
        "Optimized",
        "PDF version",
    ]

    cmd_output = subprocess.check_output([cmd, infile])
    for line in map(str, cmd_output.splitlines()):
        for label in labels:
            if label in line:
                output[label] = _extract(line)

    return output


def group_sections(chunk_nodes: List, chunk_len: int, text_splitter) -> List:
    grouped_sections = []
    current_group = []
    current_length = 0

    for node in chunk_nodes:
        if hasattr(node, "visited"):
            continue

        node_length = text_splitter.count_tokens(
            text=node.to_text(include_children=True, recurse=True)
        )

        def node_visitor(input):
            input.visited = True

        node.iter_children(node, 0, node_visitor)
        if current_length + node_length > chunk_len:
            grouped_sections.append(current_group)
            current_group = []
            current_length = 0

        current_group.append(node)
        current_length += node_length

    if current_group:
        grouped_sections.append(current_group)

    return grouped_sections


def children_ids_get(node):
    children_ids = []

    if isinstance(node, str):
        return children_ids

    for n_children in node.children:
        children_ids.append(n_children._id)

    return children_ids


def recursively_mark_nodes(node, parent, id_to_node_map):
    if hasattr(node, "id"):
        return

    node._id = str(uuid.uuid4())
    node.parent_id = parent
    id_to_node_map[node._id] = node

    for n_children in node.children:
        recursively_mark_nodes(n_children, node._id, id_to_node_map=id_to_node_map)


def assign_ids_to_doc_tree(nlmatics_doc, id_to_node_map):
    root_id = str(uuid.uuid4())
    nlmatics_doc.root_node._id = root_id
    id_to_node_map[root_id] = nlmatics_doc

    for node in nlmatics_doc.root_node.children:
        recursively_mark_nodes(
            node, nlmatics_doc.root_node._id, id_to_node_map=id_to_node_map
        )


def write_document_text(nlmatics_doc, base_dir):
    document_text = ""
    for node in nlmatics_doc.root_node.children:
        text = node.to_text(include_children=True, recurse=True)
        document_text += (
            text.encode("ascii", "ignore").decode("utf-8")
            if isinstance(text, bytes)
            else text
        )

    output_file_path = base_dir / "document.txt"
    with open(f"{output_file_path}", "w") as file:
        file.write(document_text)


# def write_sections_to_file(section_nodes, base_dir):
#     sections = []
#     for node in section_nodes:
#         s = {
#             "id": node._id,
#             "parent_id": "" if isinstance(node, Document) else node.parent_id,
#             "text": node.to_text().encode("ascii", "ignore").decode("utf-8"),
#             "tag": node.tag if hasattr(node, "tag") else "",
#             "level": node.tag if hasattr(node, "level") else "",
#             "page_idx": node.page_idx if hasattr(node, "page_idx") else "",
#             "block_idx": node.block_idx if hasattr(node, "block_idx") else "",
#             "is_section": "yes" if isinstance(node, Section) else "no",
#             "title": node.title if isinstance(node, Section) else "",
#             "block_class": (node.block_class if hasattr(node, "block_class") else ""),
#         }
#         sections.append(s)

#     output_file_path = base_dir / "section_nodes.json"
#     with open(f"{output_file_path}", "w") as file:
#         file.write(json.dumps(sections, indent=4))


def get_surrounding_section_texts(
    previous_section_grp_ids,
    next_section_grp_ids,
    current_grp,
    max_section_size,
    token_text_splitter,
    id_to_node_map,
):
    section_text = ""

    for node in current_grp:
        text = node.to_text(include_children=True, recurse=True)
        section_text += (
            text.encode("ascii", "ignore").decode("utf-8")
            if isinstance(text, bytes)
            else text
        ) + "\n"

    current_text = section_text

    # (Rest of the function remains unchanged)


    while (
        token_text_splitter.count_tokens(text=current_text) < max_section_size
        and previous_section_grp_ids
        or next_section_grp_ids
    ):
        section_text = current_text

        if previous_section_grp_ids:
            prev_section_id = previous_section_grp_ids.pop()
            current_text = (
                id_to_node_map[prev_section_id]
                .to_text(include_children=True, recurse=True)
                .encode("ascii", "ignore")
                .decode("utf-8")
            ) + current_text

        if token_text_splitter.count_tokens(text=current_text) > max_section_size:
            break

        if next_section_grp_ids:
            next_section_id = next_section_grp_ids.pop(0)
            current_text += (
                id_to_node_map[next_section_id]
                .to_text(include_children=True, recurse=True)
                .encode("ascii", "ignore")
                .decode("utf-8")
            )

        if token_text_splitter.count_tokens(text=current_text) > max_section_size:
            break

    print(
        f"section_text: {len(section_text)} tokens: {token_text_splitter.count_tokens(text=section_text)} max_section_size: {max_section_size}"
    )
    return section_text


def get_prev_section_ids(grouped_sections, group_idx):
    ids = []
    for grp in grouped_sections[:group_idx]:
        for section in grp:
            ids.append(section._id)

    return ids


def get_next_section_ids(grouped_sections, group_idx):
    ids = []
    for grp in grouped_sections[group_idx:]:
        for section in grp:
            ids.append(section._id)

    return ids


def group_sections_by_tokens(chunk_nodes: List, text_splitter, max_section_token_len) -> List:
    grouped_sections = []
    current_group = []
    current_length = 0

    sections = []

    for idx, node in enumerate(chunk_nodes):
        if hasattr(node, "visited"):
            continue

        if chunk_nodes[idx].tag != "header":
            continue

        token_length = text_splitter.count_tokens(
            text=node.to_text(include_children=True, recurse=True)
        )

        def node_visitor(input):
            input.visited = True

        node.iter_children(node, 0, node_visitor)
        if current_length + token_length > max_section_token_len:
            grouped_sections.append(current_group)
            current_group = []
            current_length = 0

        current_group.append(node)
        current_length += token_length

    if current_group:
        grouped_sections.append(current_group)

    for idx, grp in enumerate(grouped_sections):
        section = {}
        for s in grp:
            text = s.to_context_text(include_section_info=True).replace("\n", " > ")
            text = (
                text.encode("ascii", "ignore").decode("utf-8")
                if isinstance(text, bytes)
                else text
            )
            section.update(
                {
                    "idx": idx,
                    "text": text,
                    "page_idx": s.page_idx if hasattr(s, "page_idx") else "",
                    "toolname": "nlmatics",
                    "token_count": text_splitter.count_tokens(text=text),
                }
            )

        sections.append(section)

    return sections



class NlmaticsExtractionFailed(Exception):
    pass


def nlmatics_process_pdf(
    doc: Path,
    base_dir: Path = None,
    nlmatics_deployment: str = "http://localhost:5010/api/parseDocument?renderFormat=all&applyOcr=yes",
    max_section_size: int = 1024,
    text_splitter = SentenceTransformersTokenTextSplitter(model_name="BAAI/bge-large-en-v1.5", tokens_per_chunk=512, chunk_overlap=50)
) -> None:
    print(f"using nlmatics doc: {doc}")

    nlmatics_url = f"{nlmatics_deployment}/api/parseDocument?renderFormat=all&useNewIndentParser=true"
    pdf_reader = LayoutPDFReader(nlmatics_url)
    nlmatics_doc = pdf_reader.read_pdf(str(doc))

    # if nlmatics.json is empty trigger unstructured processing
    if not nlmatics_doc.json:
        raise NlmaticsExtractionFailed(
            "Empty nlmatics json, trigger unstructured processing"
        )

    output_file_path = base_dir / "doc.json"
    with open(f"{output_file_path}", "w") as file:
        file.write(json.dumps(nlmatics_doc.json, indent=4))

    id_to_node_map = {}
    assign_ids_to_doc_tree(nlmatics_doc, id_to_node_map)
    write_document_text(nlmatics_doc, base_dir)

    section_nodes = []
    for section in nlmatics_doc.sections(): 
        section_text = ""
        for child in section.children:
            if child.tag != "header":
                section_text += (
                    child.to_text(include_children=True, recurse=True)
                    .encode("ascii", "ignore")
                    .decode("utf-8")
                    + "\n"
                )

        section_nodes.append(
            {
                "id": section._id if hasattr(section, "_id") else "",
                "parent_id": section.parent_id,
                "docname": Path(doc).name,
                "text": section.to_text().encode("ascii", "ignore").decode("utf-8"),
                "tag": section.tag if hasattr(section, "tag") else "",
                "level": section.tag if hasattr(section, "level") else "",
                "page_idx": (
                    section.page_idx if hasattr(section, "page_idx") else ""
                ),
                "section title": section.title if hasattr(section, "title") else "",
                "context": section.to_context_text(include_section_info=True)
                .encode("ascii", "ignore")
                .decode("utf-8"),
                "section_text": section_text,
                "token_count": text_splitter.count_tokens(text=section_text),
                "token_count": 0,
                "block_class": (
                    section.block_class if hasattr(section, "block_class") else ""
                ),
            }
        )

    output_file_path = base_dir / "sections.json"
    with open(f"{output_file_path}", "w") as file:
        file.write(json.dumps(section_nodes, indent=4))

    chunk_nodes = []
    for node in nlmatics_doc.root_node.children:
        def recursively_add_nodes(n):
            if (
                text_splitter.count_tokens(
                    text=n.to_text(include_children=True, recurse=True)
                )
                < max_section_size * 0.95  
            ):
                chunk_nodes.append(n)
                return

            for n_children in n.children:
                recursively_add_nodes(n_children)
        recursively_add_nodes(node)

    # Group the sections based on the parent id
    grouped_sections = group_sections_by_tokens(
        chunk_nodes, text_splitter=text_splitter, max_section_token_len=max_section_size
    )
    output_file_path = base_dir / "grouped_sections.json"
    with open(f"{output_file_path}", "w") as file:
        file.write(json.dumps(grouped_sections, indent=4))

    # Write the nodes to a json file
    doc_nodes = []
    node_idx = 0
    for _id, node in id_to_node_map.items():
        if isinstance(node, Document):
            continue

        doc_nodes.append(
            {
                "id": _id,
                "idx": node_idx,
                "parent_id": "" if isinstance(node, Document) else node.parent_id,
                "text": node.to_text().encode("ascii", "ignore").decode("utf-8"),
                "tag": node.tag if hasattr(node, "tag") else "",
                "level": node.tag if hasattr(node, "level") else "",
                "page_idx": node.page_idx if hasattr(node, "page_idx") else "",
                "block_idx": node.block_idx if hasattr(node, "block_idx") else "",
                "is_section": "yes" if isinstance(node, Section) else "no",
                "title" : node.title if isinstance(node, Section) else "",
                "block_class": (
                    node.block_class if hasattr(node, "block_class") else ""
                ),
            }
        )
        node_idx += 1

    output_file_path = base_dir / "nlmatics_nodes.json"
    with open(f"{output_file_path}", "w") as file:
        file.write(json.dumps(doc_nodes, indent=4))

    tool_file_path = base_dir / "nlmatics.touch"
    tool_file_path.touch()


# used for pdf extraction unstructured
def unstructured_process(
            path: Path,
            base_dir: Path = None,
    ) -> None:
    try:
        print(f"Unstructured Processing file: {path} basedir: {base_dir}")
        path = Path(path)

        if path.suffix == ".pdf":
            elements = partition_pdf(filename=str(path.absolute()),
                                        infer_table_structure=True,
                                        strategy="hi_res",
                                        hi_res_model_name="yolox")

        elif path.suffix == ".docx":
            elements = partition_docx(filename=str(
                path.absolute()), infer_table_structure=True)

        elif path.suffix == ".pptx":
            elements = partition_pptx(filename=str(
                path.absolute()), infer_table_structure=True)

        page_dict = {}
        for el in elements:
            el_pg_no = el.metadata.page_number
            if el_pg_no not in page_dict:
                page_dict[el.metadata.page_number] = {
                    'page_text': '',
                    'tables': [],
                    'is_table': False,
                    'is_pdf': True,
                    "toolname" : "unstructured",
                    'docname': Path(path).name
                }

            page_dict[el_pg_no]['page_no'] = el_pg_no
            if el.category == "Table":
                page_dict[el_pg_no]['tables'].append(
                    el.metadata.text_as_html)
                page_dict[el_pg_no]['is_table'] = True
                page_dict[el_pg_no]['page_text'] += f"{el.metadata.text_as_html}\n"
            else:
                page_dict[el_pg_no]['page_text'] += f"{el.text}\n"

        for page, content in page_dict.items():
            output_dir = base_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_dir / f"page_{page}.json"

            with open(f"{output_file_path}", 'w') as file:
                file.write(json.dumps(content, indent=4))

    except Exception as e:
        print(f"Error in unstructured_process_pdf: {e}")
        traceback.print_exc()


def unstructured_process_pptx(path: Path, base_dir: Path = None) -> None:
    try:
        print(f"Processing file: {path} basedir: {base_dir}")
        path = Path(path)

        elements = partition_pptx(
            filename=path, infer_table_structure=True)
        page_dict = {}
        for el in elements:
            el_pg_no = el.metadata.page_number
            if el_pg_no not in page_dict:
                page_dict[el.metadata.page_number] = {
                    'page_text': '',
                    'tables': [],
                    'is_table': False,
                    'is_pdf': True,
                    'docname': Path(path).name
                }

            page_dict[el_pg_no]['page_no'] = el_pg_no
            if el.category == "Table":
                page_dict[el_pg_no]['tables'].append(
                    el.metadata.text_as_html)
                page_dict[el_pg_no]['is_table'] = True
                page_dict[el_pg_no]['page_text'] += f"{el.metadata.text_as_html}\n"
            else:
                page_dict[el_pg_no]['page_text'] += f"{el.text}\n"

        filename = path.name
        for page, content in page_dict.items():
            output_dir = base_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_dir / f"page_{page}.json"

            with open(f"{output_file_path}", 'w') as file:
                file.write(json.dumps(content, indent=4))

    except Exception as e:
        print(f"Error in unstructured_process_pdf: {e}")
        traceback.print_exc()


def process_faqs(
    path: Path, 
    col1: int = 0, 
    col2: int = 1, 
    base_dir: Path = None
    ) -> None:
    try:
        path = Path(path)
        excel_data = pd.read_excel(path, sheet_name=None)

        for sheet_name, sheet_data in excel_data.items():
            sheet_output_dir = base_dir / sheet_name
            sheet_output_dir.mkdir(parents=True, exist_ok=True)

            for index, row in sheet_data.iterrows():
                question = row.iloc[col1] if col1 < len(row) else ""
                answer = row.iloc[col2] if col2 < len(row) else ""
                text = "{}\n    \n    \n{}".format(question, answer)

                qa_pair = {
                    "url": path.name,
                    "question": question,
                    "answer": answer,
                    "text": text,
                }

                output_json_path = sheet_output_dir / f"faq_{index + 1}.json"

                with open(output_json_path, 'w') as file:
                    file.write(json.dumps(qa_pair, indent=4))

    except Exception as e:
        print(f"Error in process_faqs: {e}")
        traceback.print_exc()


def add_cache_match_to_sllm(query, answer, trace_id, llmkey, model_name, user):
    supabase_kong_url = "http://supabase-kong.securellm:8000"
    supabase_rest_url = f"{supabase_kong_url}/rest/v1"

    supabaseAnnonKey = os.getenv("SUPABASEANNONKEY", None)
    supabaseServiceKey = os.getenv("SUPABASESERVICEKEY", None)
    
    if supabaseAnnonKey == None:
        raise ValueError("Error: Did not find the SUPABASEANNONKEY environment variable.")
    
    if supabaseServiceKey == None:
        raise ValueError("Error: Did not fine the SUPABASESERVICEKEY environment variable.")

    headersList = {
        "Accept": "*/*",
        "apikey": supabaseServiceKey,
        "Content-Type": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {supabaseServiceKey}"
    }

    def hash(key: str) -> str:
        encoded_key = key.encode("utf-8")
        hex_codes = [format(b, "02x") for b in encoded_key]
        return "".join(hex_codes)

    def _get_appkey_byhash(h_appkey):
        result = (
            dbclient.from_("app_keys").select("*").eq("api_key_hash", h_appkey).execute()
        )

        if len(result.data) == 0:
            return None
        return result.data[0]["api_key_name"].split("@")[2]

    def insert_request(data):
        payload = json.dumps(data)
        reqUrl = f"{supabase_rest_url}/request"
        response = requests.request(
            "POST", reqUrl, data=payload,  headers=headersList, verify=False)
        if response.status_code == 200:
            resData = response.json()
        return None


    def insert_response(data):
        payload = json.dumps(data)
        reqUrl = f"{supabase_rest_url}/response"
        response = requests.request(
            "POST", reqUrl, data=payload,  headers=headersList, verify=False)
        if response.status_code == 200:
            resData = response.json()
        return None


    def get_application_id(appKeyHash):
        # payload = json.dumps(data)
        api_key_name = None
        reqUrl = f"{supabase_rest_url}/app_keys"
        response = requests.request(
            "GET", reqUrl,  headers=headersList, verify=False)
        
        if response.status_code == 200:
            resData = response.json()
            
            i = 0
            while(i < len(resData) and api_key_name == None):
                if resData[i]['api_key_hash'] == appKeyHash:
                    api_key_name = resData[i]['api_key_name']
                i = i + 1

        return api_key_name
  

    def get_llmkey_hash(provider = "openai"):
        reqUrl = f"{supabase_rest_url}/llm_keys"
        response = requests.request(
            "GET", reqUrl,  headers=headersList, verify=False)
        if response.status_code == 200:
            resData = response.json()
            return resData[0]['api_key_hash']
        return None

    id = str(uuid.uuid4())
    flow_id = str(trace_id) # replace the value with id which is coming from the securechat app
    hash_value = hash(f'Bearer {llmkey}')
    application_id = get_application_id(hash_value)
    
    if application_id == None:
        raise ValueError("Error: Invalid Openai api key for securellm logging.")

    llmkey_hash = get_llmkey_hash()
    requestBody = {
        "input": query,
        "model": model_name,
        "application_id": application_id
    }

    requestData = {
        "id": id,
        "body": requestBody,
        "path": "/v1/chat/completions",
        "auth_hash": llmkey_hash,
        "user_id": user,
        "properties": {
            "cost": 0,
            "flow_id": flow_id,
            "cached": True,
            "application_key_hash": hash_value,
        }
    }

    responseBody = {
        "model": model_name,
        "usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer 
                },
                "finish_reason": "stop"
            }
        ]
    }

    responseData = {
        "id": id,
        "body": responseBody,
        "request": id,
        "status": 200,
        "completion_tokens": 0,
        "prompt_tokens": 0,
    }

    insert_request(requestData)
    insert_response(responseData)
    
    
    
    
def main():
    # Set up paths and variables
    document_path = Path("/home/ocdlgit/Data/NVHL/NVHL_2023_Policy_Manual_v5.1-Final-Clean-Version.pdf")
    base_dir = Path("/home/ocdlgit/output_fmutils")
    nlmatics_deployment = "http://localhost:5010"  # Assuming the Docker container is running
    max_section_size = 1024

    # Ensure the output directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    # Process the document using nlmatics
    try:
        nlmatics_process_pdf(
            doc=document_path,
            base_dir=base_dir,
            nlmatics_deployment=nlmatics_deployment,
            max_section_size=max_section_size,
            text_splitter = SentenceTransformersTokenTextSplitter(model_name="BAAI/bge-large-en-v1.5", tokens_per_chunk=512, chunk_overlap=50)
        )
        print("Document processed successfully. Output stored in:", base_dir)
    except NlmaticsExtractionFailed as e:
        print(f"Nlmatics processing failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
