import os
import requests
import unicodedata

# 1. SETUP OUTPUT
output_dir = "./nigerian_txt_datasets"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. UPDATED DIRECT GITHUB LINKS (MasakhaNER 2.0 Raw Data)
# These links point to the 'train.txt' files for each language
lang_urls = {
    "yoruba": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/master/MasakhaNER2.0/data/yor/train.txt",
    "hausa": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/master/MasakhaNER2.0/data/hau/train.txt",
    "igbo": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/master/MasakhaNER2.0/data/ibo/train.txt",
    "pidgin": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/master/MasakhaNER2.0/data/pcm/train.txt"
}

def extract_raw_text():
    print("🚀 Initializing Neural Signal Stream (MasakhaNER 2.0)...")
    
    for lang, url in lang_urls.items():
        print(f"⏳ Syncing {lang.upper()} data...")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                lines = response.text.splitlines()
                clean_sentences = []
                current_sentence = []
                
                # Processing CoNLL format: Word <TAB> Label
                for line in lines:
                    if line.strip() == "": # End of sentence
                        if current_sentence:
                            clean_sentences.append(" ".join(current_sentence))
                            current_sentence = []
                    else:
                        # Extract the word (first column)
                        parts = line.split()
                        if len(parts) >= 1:
                            word = parts[0]
                            current_sentence.append(word)
                
                # Final normalization for Yoruba/Igbo tonal marks
                final_text = unicodedata.normalize('NFC', "\n".join(clean_sentences))
                
                file_path = os.path.join(output_dir, f"{lang}_dataset.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
                
                print(f"✅ Success: {lang} -> {len(clean_sentences)} sentences extracted.")
            else:
                print(f"❌ 404 Error: Could not reach {lang} via GitHub API.")
                
        except Exception as e:
            print(f"❌ Error on {lang}: {e}")

if __name__ == "__main__":
    extract_raw_text()
    print("\n🏁 Your Nigerian datasets are ready in VS Code!")