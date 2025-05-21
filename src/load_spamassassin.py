import os, email, pandas as pd

def parse_spamassassin(base_path='dataset/spamAssassin'):
    rows = []
    # Walk every subdirectory under the raw folder
    for root, dirs, files in os.walk(base_path):
        for fname in files:
            path = os.path.join(root, fname)

            # Determine ham vs spam by folder name in the path
            low = root.lower()
            if 'easy_ham' in low:
                label = 'ham'
            elif 'spam' in low:
                label = 'spam'
            else:
                # skip any unrelated files
                continue

            # Try to parse as an email
            try:
                with open(path, 'rb') as f:
                    msg = email.message_from_binary_file(f)
            except Exception:
                continue
            
            frm   = str(msg.get('From','')).strip()
            subj  = str(msg.get('Subject','')).strip()
            date  = str(msg.get('Date','')).strip()


            # Pull out only the text/plain parts
            if msg.is_multipart():
                parts = []
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain' and not part.get('Content-Disposition'):
                        parts.append(part.get_payload(decode=True) or b'')
                body = b''.join(parts).decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True) or b''
                body = body.decode('utf-8', errors='ignore')

            rows.append({
                'label':   label,
                'from':    frm,
                'subject': subj,
                'date':    date,
                'body':    body
            })

    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = parse_spamassassin()
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/spamassassin_raw.csv', index=False)
    print(f" Parsed {len(df)} emails to data/spamassassin_raw.csv")
