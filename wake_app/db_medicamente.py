import pandas as pd
import unicodedata
import os
import re

def strip_accents(text: str) -> str:
    """
    Elimină diacriticele și pune textul pe lowercase.
    Exemplu: "Paracetamol ProBiotic" → "paracetamol probiotic"
    """
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

def extract_base_name(full_name: str) -> str:
    """
    Dintr-o denumire completă ca "PARACETAMOL 500 mg" 
    sau "Paracetamol Acetaminofen 10 mg/ml", 
    returnează doar partea alfabetică de la început, 
    înainte să apară prima cifră. Rezultatul e lowercase și fără diacritice.
    
    Exemple:
      "PARACETAMOL 500 mg"     -> "paracetamol"
      "Ibuprofen 200mg compr." -> "ibuprofen"
      "Paracetamol Accord 10 mg/ml" -> "paracetamol accord"
    """
    if not isinstance(full_name, str):
        return ""
    # Mai întâi normalizăm (fără diacritice, lowercase)
    norm = strip_accents(full_name)
    # Luăm tot până la prima cifră (0–9). Dacă nu găsește cifră, întoarce tot șirul.
    match = re.match(r"^([a-z\s]+?)(?=\s*\d)", norm)
    if match:
        return match.group(1).strip()
    else:
        # Dacă nu există cifră în text, luăm tot
        return norm.strip()

class MedDatabase:
    """
    Încarcă 'medicamente.xlsx' și pune datele într-o listă de dicționare.
    Oferă:
      - search_by_prefix(prefix) pentru lookup simplu după denumire.
      - get_base_name(med) pentru a extrage doar denumirea de bază.
    """
    def __init__(self, excel_path="/home/user/Desktop/Internship-Siemens/wake_app/medicamente.xlsx"):
        self.medicamente = []
        self._load_from_excel(excel_path)

    def _load_from_excel(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fișierul Excel '{path}' nu există.")
        df = pd.read_excel(path, engine="openpyxl")

        # Ajustează numele coloanelor după cum apar în fișierul tău:
        col_denumire  = "Denumire comerciala"   # în loc de "Denumire"
        col_substanta = "DCI"                   # în loc de "Substanța activă"
        col_forma     = "Forma farmaceutica"    # în loc de "Formă farmaceutică"
        col_dozaj     = "Concentratie"          # în loc de "Dozaj"


        # Verifică coloanele obligatorii:
        for col in [col_denumire, col_substanta, col_forma, col_dozaj]:
            if col not in df.columns:
                raise KeyError(f"Coloana '{col}' nu există în '{path}'. Verifică numele exact al coloanei.")

        for _, row in df.iterrows():
            denumire = str(row[col_denumire]).strip()
            if not denumire or denumire.lower() == "nan":
                continue

            substanta = str(row[col_substanta]).strip() if col_substanta in df.columns else ""
            forma      = str(row[col_forma]).strip() if col_forma in df.columns else ""
            dozaj      = str(row[col_dozaj]).strip() if col_dozaj in df.columns else ""
            prospect   = ""
          
            norm_name = strip_accents(denumire)

            med = {
                "denumire": denumire,
                "substanta_activa": substanta,
                "forma_farmaceutica": forma,
                "dozaj": dozaj,
                "prospect_url": prospect,
                "denumire_norm": norm_name,
            }
            self.medicamente.append(med)

    def search_by_prefix(self, prefix: str, limit=5):
        """
        Caută medicamente a căror 'denumire_norm' începe cu prefix_norm.
        Returnează maxim `limit` rezultate (listă de dicționare).
        Dacă prefix e gol sau None, întoarce listă goală.
        """
        if not prefix or not isinstance(prefix, str):
            return []

        prefix_norm = strip_accents(prefix)
        rezultate = []
        for med in self.medicamente:
            if med["denumire_norm"].startswith(prefix_norm):
                rezultate.append(med)
                if len(rezultate) >= limit:
                    break
        return rezultate

    def get_base_name(self, med: dict) -> str:
        """
        Dintr-un dicționar 'med' (având cheia 'denumire' sau 'denumire_norm'),
        extrage doar denumirea de bază (fără dozaj, fără cifre).
        Se folosește funcția extract_base_name pe 'denumire'.
        """
        full_name = med.get("denumire", "")
        return extract_base_name(full_name)

# -------------------------------
# Exemplu de utilizare în aplicație:

if __name__ == "__main__":
    # Încarcă baza de date
    db = MedDatabase(excel_path="/home/user/Desktop/Internship-Siemens/wake_app/medicamente.xlsx")

    # Căutăm toate medicamentele care încep cu "par"
    rezultate = db.search_by_prefix("aspi", limit=5)
    print("Rezultate brute (5):")
    for med in rezultate:
        print("  -", med["denumire"])

    # Extragem doar numele de bază și îl afișăm
    print("\nDenumiri de bază extrase:")
    for med in rezultate:
        base = db.get_base_name(med)
        print("  -", base)

    # Exemplu simplu de „rostire” (just pseudo-cod)
    # Presupunem că ai un modul tts care are o funcție say(text)
    from tts_module import say
    
    for med in rezultate:
        base = db.get_base_name(med)
        say(base) 