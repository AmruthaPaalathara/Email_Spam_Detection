import os, email, pandas as pd  # os for file walking, email for parsing, pandas for DataFrame

def parse_spamassassin(base_path='dataset/spamAssassin'):
    rows = []  # collect one dict per email

    # Walk every subdirectory under base_path
    for root, dirs, files in os.walk(base_path):
        for fname in files:
            path = os.path.join(root, fname)  # full path to this file

            # Label by folder name
            low = root.lower()  
            if 'easy_ham' in low:
                label = 'ham'  
            elif 'spam' in low:
                label = 'spam'  
            else:
                continue  # skip files not under easy_ham or spam

            # Try to parse the file as an email
            try:
                with open(path, 'rb') as f:
                    msg = email.message_from_binary_file(f)  # binary-safe parsing
            except Exception:
                continue  # skip if not a valid email

            # Extract header fields safely
            frm   = str(msg.get('From','')).strip()    # sender
            subj  = str(msg.get('Subject','')).strip() # subject line
            date  = str(msg.get('Date','')).strip()    # date header

            # Extract only the text/plain parts of the body
            if msg.is_multipart():
                parts = []
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain' and not part.get('Content-Disposition'):
                        parts.append(part.get_payload(decode=True) or b'')
                body = b''.join(parts).decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True) or b''  # single-part payload
                body = body.decode('utf-8', errors='ignore')

            # Append a record for this email
            rows.append({
                'label':   label,
                'from':    frm,
                'subject': subj,
                'date':    date,
                'body':    body
            })

    return pd.DataFrame(rows)  # convert list of dicts → DataFrame

if __name__ == '__main__':
    df = parse_spamassassin()                         # parse all emails
    os.makedirs('data', exist_ok=True)                # ensure output folder exists
    df.to_csv('data/spamassassin_raw.csv', index=False)  # save to CSV
    print(f" Parsed {len(df)} emails to data/spamassassin_raw.csv")
