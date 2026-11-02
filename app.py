"""
DeepProtein Studio - Production Inference Server
Serves the custom DeepProteinHybrid model via FastAPI
"""

import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from train_protein_pipeline import DeepProteinHybrid, VOCAB, PHYSICO_MATRIX

app = FastAPI(title="DeepProtein Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# 1. 40 GO TERMS COMPREHENSIVE METADATA
# ==================================================
GO_METADATA = {
    'GO:0005737': {'name': 'Cytoplasm', 'domain': 'CC', 'desc': 'The contents of a cell contained within the plasma membrane, excluding the nucleus.'},
    'GO:0005634': {'name': 'Nucleus', 'domain': 'CC', 'desc': 'A membrane-bounded eukaryotic organelle containing genetic chromosomes.'},
    'GO:0005829': {'name': 'Cytosol', 'domain': 'CC', 'desc': 'The fluid portion of the cytoplasm excluding membrane-bound organelles.'},
    'GO:0005886': {'name': 'Plasma Membrane', 'domain': 'CC', 'desc': 'The membrane separating the cell cytoplasm from the extracellular space.'},
    'GO:0005524': {'name': 'ATP Binding', 'domain': 'MF', 'desc': 'Selective, non-covalent interaction with adenosine triphosphate (ATP).'},
    'GO:0005576': {'name': 'Extracellular Region', 'domain': 'CC', 'desc': 'The outer space exterior to the outermost boundary of a cell.'},
    'GO:0016020': {'name': 'Membrane', 'domain': 'CC', 'desc': 'Lipid bilayer structure enclosing cells or cellular sub-compartments.'},
    'GO:0008270': {'name': 'Zinc Ion Binding', 'domain': 'MF', 'desc': 'Interacting selectively and non-covalently with zinc (Zn2+) ions.'},
    'GO:0042802': {'name': 'Identical Protein Binding', 'domain': 'MF', 'desc': 'Interacting with another molecule of the identical protein to form homomers.'},
    'GO:0000287': {'name': 'Magnesium Ion Binding', 'domain': 'MF', 'desc': 'Interacting selectively and non-covalently with magnesium (Mg2+) ions.'},
    'GO:0005739': {'name': 'Mitochondrion', 'domain': 'CC', 'desc': 'Semiautonomous organelle responsible for aerobic cellular respiration.'},
    'GO:0005654': {'name': 'Nucleoplasm', 'domain': 'CC', 'desc': 'The viscous fluid space enclosed by the nuclear envelope.'},
    'GO:0005783': {'name': 'Endoplasmic Reticulum', 'domain': 'CC', 'desc': 'Membrane network involved in protein synthesis, folding, and lipid synthesis.'},
    'GO:0003723': {'name': 'RNA Binding', 'domain': 'MF', 'desc': 'Interacting selectively with ribonucleic acid (RNA) sequences.'},
    'GO:0003677': {'name': 'DNA Binding', 'domain': 'MF', 'desc': 'Interacting selectively with deoxyribonucleic acid (DNA) chromosomes.'},
    'GO:0006357': {'name': 'Reg. of Transcription (RNAP II)', 'domain': 'BP', 'desc': 'Modulating the frequency or rate of RNA polymerase II transcription.'},
    'GO:0005789': {'name': 'Endoplasmic Reticulum Membrane', 'domain': 'CC', 'desc': 'The lipid bilayer bounding the endoplasmic reticulum network.'},
    'GO:0000978': {'name': 'RNAP II Promoter Binding', 'domain': 'MF', 'desc': 'Sequence-specific binding to core or proximal promoter DNA regions.'},
    'GO:0042803': {'name': 'Protein Homodimerization Activity', 'domain': 'MF', 'desc': 'Interacting with an identical protein molecule to form a functional dimer.'},
    'GO:0000981': {'name': 'Transcription Factor Activity (RNAP II)', 'domain': 'MF', 'desc': 'Sequence-specific transcription factor activating RNA polymerase II.'},
    'GO:0003700': {'name': 'DNA-Binding Transcription Factor', 'domain': 'MF', 'desc': 'Regulates transcription of specific genes by binding target DNA motifs.'},
    'GO:0016887': {'name': 'ATPase Activity', 'domain': 'MF', 'desc': 'Catalysis of the hydrolysis of ATP to ADP and inorganic phosphate.'},
    'GO:0005730': {'name': 'Nucleolus', 'domain': 'CC', 'desc': 'Dense subnuclear structure where ribosomes and rRNAs are synthesized.'},
    'GO:0009507': {'name': 'Chloroplast', 'domain': 'CC', 'desc': 'Photosynthetic plastid organelle found in plant and algal cells.'},
    'GO:0045944': {'name': 'Positive Reg. of Transcription (RNAP II)', 'domain': 'BP', 'desc': 'Activating or increasing transcription frequency by RNA polymerase II.'},
    'GO:0005794': {'name': 'Golgi Apparatus', 'domain': 'CC', 'desc': 'Organelle that packages and modifies proteins for secretion or cellular targeting.'},
    'GO:0006355': {'name': 'Regulation of Transcription, DNA-templated', 'domain': 'BP', 'desc': 'Controlling the synthesis of RNA molecules from DNA templates.'},
    'GO:0005743': {'name': 'Mitochondrial Inner Membrane', 'domain': 'CC', 'desc': 'The inner lipid bilayer of the mitochondrion containing electron transport cristae.'},
    'GO:0020037': {'name': 'Heme Binding', 'domain': 'MF', 'desc': 'Interacting selectively with a heme cofactor (iron-protoporphyrin complex).'},
    'GO:0046872': {'name': 'Metal Ion Binding', 'domain': 'MF', 'desc': 'Interacting selectively with any mono- or polyvalent metal ion.'},
    'GO:0005509': {'name': 'Calcium Ion Binding', 'domain': 'MF', 'desc': 'Interacting selectively and non-covalently with calcium (Ca2+) ions.'},
    'GO:0005525': {'name': 'GTP Binding', 'domain': 'MF', 'desc': 'Selective, non-covalent interaction with guanosine triphosphate (GTP).'},
    'GO:0006508': {'name': 'Proteolysis', 'domain': 'BP', 'desc': 'Enzymatic cleavage of peptide bonds in protein substrates.'},
    'GO:0000139': {'name': 'Golgi Membrane', 'domain': 'CC', 'desc': 'The lipid bilayer bounding the cisternae of the Golgi apparatus.'},
    'GO:0051539': {'name': '4Fe-4S Cluster Binding', 'domain': 'MF', 'desc': 'Interacting selectively with an iron-sulfur [4Fe-4S] prosthetic cluster.'},
    'GO:0005975': {'name': 'Carbohydrate Metabolic Process', 'domain': 'BP', 'desc': 'Cellular chemical reactions involving carbohydrates like sugars and glycogen.'},
    'GO:0005506': {'name': 'Iron Ion Binding', 'domain': 'MF', 'desc': 'Interacting selectively with iron (Fe2+ or Fe3+) ions.'},
    'GO:0009986': {'name': 'Cell Surface', 'domain': 'CC', 'desc': 'The external layer of the cell encompassing surface receptors and adhesion molecules.'},
    'GO:0106310': {'name': 'Protein Serine Kinase Activity', 'domain': 'MF', 'desc': 'Catalysis of the transfer of a phosphate group onto target serine residues.'},
    'GO:0048471': {'name': 'Perinuclear Region of Cytoplasm', 'domain': 'CC', 'desc': 'The region of cytoplasm immediately surrounding the cell nucleus.'}
}

DOMAIN_NAMES = {
    'MF': 'Molecular Function',
    'BP': 'Biological Process',
    'CC': 'Cellular Component'
}

# ==================================================
# 2. LOAD TRAINED PRODUCTION MODEL
# ==================================================
MODEL_PATH = "deepprotein_hybrid_production.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Please train model or save checkpoint.")

print(f"Loading DeepProteinHybrid production weights from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

CLASSES = checkpoint["classes"]
PER_CLASS_THRESHOLDS = checkpoint.get("per_class_thresholds", [0.35] * len(CLASSES))
if isinstance(PER_CLASS_THRESHOLDS, (list, tuple)):
    PER_CLASS_THRESHOLDS = {c: float(t) for c, t in zip(CLASSES, PER_CLASS_THRESHOLDS)}
elif hasattr(PER_CLASS_THRESHOLDS, "__len__"):
    PER_CLASS_THRESHOLDS = {c: float(t) for c, t in zip(CLASSES, PER_CLASS_THRESHOLDS)}

model = DeepProteinHybrid(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"DeepProteinHybrid loaded successfully on {DEVICE} for {len(CLASSES)} GO classes!")

# ==================================================
# 3. BIOPHYSICAL ANALYSIS UTILITIES
# ==================================================
AA_MW = {
    'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
}

KYTE_DOOLITTLE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
    'W': -0.9, 'Y': -1.3
}

def compute_biophysics(seq: str) -> Dict[str, Any]:
    length = len(seq)
    if length == 0:
        return {}
    # Molecular weight (sum of residues minus water per peptide bond)
    raw_mw = sum(AA_MW.get(aa, 110.0) for aa in seq)
    mw_kda = round((raw_mw - (length - 1) * 18.015) / 1000.0, 2)
    # Net charge at pH 7.4
    pos = seq.count('K') + seq.count('R') + 0.1 * seq.count('H')
    neg = seq.count('D') + seq.count('E')
    net_charge = round(pos - neg, 1)
    # Hydropathy (GRAVY)
    gravy = round(sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / length, 2)
    # Approximate Isoelectric Point (pI)
    pi_estimate = round(6.0 + (net_charge / max(1, length)) * 10.0, 2)
    pi_estimate = max(3.5, min(11.5, pi_estimate))
    
    return {
        "length": length,
        "molecular_weight_kda": mw_kda,
        "isoelectric_point": pi_estimate,
        "net_charge_ph7": net_charge,
        "gravy_hydropathy": gravy,
        "hydrophobic_pct": round(sum(1 for aa in seq if aa in 'AILMFWV') / length * 100, 1),
        "polar_pct": round(sum(1 for aa in seq if aa in 'STNQCY') / length * 100, 1),
        "charged_pct": round(sum(1 for aa in seq if aa in 'KRHDE') / length * 100, 1)
    }

# ==================================================
# 4. PRESET BENCHMARK PROTEINS
# ==================================================
PRESET_SAMPLES = [
    {
        "id": "INSULIN",
        "name": "Human Insulin (P01308)",
        "organism": "Homo sapiens",
        "desc": "Key hormone regulating carbohydrate and lipid metabolism. Binds insulin receptor to facilitate glucose uptake.",
        "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    },
    {
        "id": "HBB",
        "name": "Hemoglobin Subunit Beta (P68871)",
        "organism": "Homo sapiens",
        "desc": "Iron-containing oxygen transport metalloprotein in erythrocytes. Contains heme cofactor binding pockets.",
        "sequence": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    },
    {
        "id": "P53",
        "name": "Cellular Tumor Antigen p53 (P04637)",
        "organism": "Homo sapiens",
        "desc": "Master sequence-specific DNA-binding transcription factor and tumor suppressor regulating cell cycle arrest and apoptosis.",
        "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    },
    {
        "id": "CYP3A4",
        "name": "Cytochrome P450 3A4 (P08684)",
        "organism": "Homo sapiens",
        "desc": "Major endoplasmic reticulum monooxygenase metabolizing approximately 50% of therapeutic drugs. Contains heme b cofactor.",
        "sequence": "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELMAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA"
    },
    {
        "id": "RHODOPSIN",
        "name": "Rhodopsin (P02699)",
        "organism": "Bos taurus",
        "desc": "7-transmembrane G-protein coupled photoreceptor in retinal rod cells converting photons into visual neurochemical signals.",
        "sequence": "MNGTEGPNFYVPFSNKTGVVRSPFEAPQYYLAEPWQFSMLAAYMFLLIMLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTPHEETNNESFVIYMFVVHFIIPLIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWLPYAGVAFYIFTHQGSDFGPIFMTIPAFFAKTSAVYNPVIYIMMNKQFRNCMVTTLCCGKNPLGDDEASTTVSKTETSQVAPA"
    }
]

# ==================================================
# 5. REQUEST & RESPONSE MODELS
# ==================================================
class PredictionRequest(BaseModel):
    sequence: str
    mode: Optional[str] = "calibrated"  # "calibrated" or "custom"
    custom_threshold: Optional[float] = 0.50

class GOTermPrediction(BaseModel):
    go_id: str
    name: str
    domain: str
    domain_name: str
    description: str
    probability: float
    threshold: float
    is_active: bool
    quickgo_url: str

class PredictionResponse(BaseModel):
    sequence_length: int
    cleaned_sequence: str
    biophysics: Dict[str, Any]
    predictions: List[GOTermPrediction]
    domain_summary: Dict[str, int]
    attention_weights: List[float]
    top_domains: Dict[str, List[GOTermPrediction]]

# ==================================================
# 6. API ENDPOINTS
# ==================================================
@app.get("/api/samples")
def get_samples():
    return PRESET_SAMPLES

@app.post("/api/predict", response_model=PredictionResponse)
def predict_protein_function(req: PredictionRequest):
    # 1. Clean and validate sequence
    raw_seq = req.sequence.strip()
    # Strip FASTA header if present
    lines = [line.strip() for line in raw_seq.split("\n") if line.strip()]
    if lines and lines[0].startswith(">"):
        clean_seq = "".join(lines[1:]).upper().replace(" ", "")
    else:
        clean_seq = "".join(lines).upper().replace(" ", "")

    if len(clean_seq) == 0:
        raise HTTPException(status_code=400, detail="Protein sequence cannot be empty.")

    # Validate amino acid characters
    invalid_chars = set(clean_seq) - set("ACDEFGHIKLMNPQRSTVWY")
    if invalid_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence contains invalid amino acid characters: {sorted(list(invalid_chars))}. Only standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY) are allowed."
        )

    if len(clean_seq) > 1000:
        clean_seq = clean_seq[:1000]

    # 2. Encode sequence into integer tensor
    seq_encoded = [VOCAB[aa] for aa in clean_seq]
    seq_tensor = torch.tensor([seq_encoded], dtype=torch.long, device=DEVICE)
    mask = (seq_tensor != 0)

    # 3. Model inference and attention extraction
    with torch.no_grad():
        # Feature extraction
        t_emb = model.token_emb(seq_tensor)
        p_emb = model.physico_emb(seq_tensor)
        emb = torch.cat([t_emb, p_emb], dim=-1).transpose(1, 2)
        h = model.input_proj(emb)
        
        b3 = model.b3(h)
        b5 = model.b5(h)
        b9 = model.b9(h)
        b15 = model.b15(h)
        merged = torch.cat([b3, b5, b9, b15], dim=1)

        conv_out = model.se(model.res_conv(merged), mask) + merged
        gru_in = conv_out.transpose(1, 2)
        gru_out, _ = model.gru(gru_in)

        # Attention weights over sequence
        att_scores = model.attn_pool.attn(gru_out).squeeze(-1)
        att_scores = att_scores.masked_fill(~mask, -1e9)
        att_weights = F.softmax(att_scores, dim=1)[0].cpu().numpy().tolist()

        # Dual pooling
        weights_expanded = F.softmax(att_scores, dim=1).unsqueeze(-1)
        att_feat = (gru_out * weights_expanded).sum(dim=1)
        gru_masked = gru_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_feat = torch.max(gru_masked, dim=1)[0]
        combined = torch.cat([att_feat, max_feat], dim=1)
        logits = model.classifier(combined)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # 4. Construct response objects
    predictions = []
    domain_summary = {"MF": 0, "BP": 0, "CC": 0}
    top_domains = {"MF": [], "BP": [], "CC": []}

    use_calibrated = (req.mode == "calibrated")
    custom_cutoff = req.custom_threshold if req.custom_threshold is not None else 0.50

    for idx, go_id in enumerate(CLASSES):
        prob = float(probs[idx])
        meta = GO_METADATA.get(go_id, {
            'name': go_id,
            'domain': 'MF',
            'desc': 'Biological gene ontology term.'
        })
        threshold = PER_CLASS_THRESHOLDS.get(go_id, 0.35) if use_calibrated else custom_cutoff
        is_active = (prob >= threshold)

        pred_obj = GOTermPrediction(
            go_id=go_id,
            name=meta['name'],
            domain=meta['domain'],
            domain_name=DOMAIN_NAMES.get(meta['domain'], 'Function'),
            description=meta['desc'],
            probability=round(prob, 4),
            threshold=round(float(threshold), 3),
            is_active=is_active,
            quickgo_url=f"https://www.ebi.ac.uk/QuickGO/term/{go_id}"
        )
        predictions.append(pred_obj)
        top_domains[meta['domain']].append(pred_obj)
        if is_active:
            domain_summary[meta['domain']] += 1

    # Sort each domain list by probability descending
    predictions.sort(key=lambda x: x.probability, reverse=True)
    for dom in top_domains:
        top_domains[dom].sort(key=lambda x: x.probability, reverse=True)

    # 5. Biophysical calculations
    biophysics = compute_biophysics(clean_seq)

    return PredictionResponse(
        sequence_length=len(clean_seq),
        cleaned_sequence=clean_seq,
        biophysics=biophysics,
        predictions=predictions,
        domain_summary=domain_summary,
        attention_weights=[round(float(w), 5) for w in att_weights],
        top_domains=top_domains
    )

# ==================================================
# 7. STATIC FILES SERVING (FRONTEND)
# ==================================================
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "DeepProtein API Running", "ui": "Place index.html in static/"})

if __name__ == "__main__":
    import uvicorn
    print("Launching DeepProtein Studio Web Application on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
