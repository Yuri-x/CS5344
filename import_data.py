from config import pgsql_conf
from koala.connector import PGSQLConnector
import os

if __name__ == '__main__':
    conn = PGSQLConnector(pgsql_conf)
    with open('exception.log', 'w') as f:
        for folder in ('tweets', 'users'):
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if os.path.isdir(path):
                    continue
                try:
                    print(f"Processing {path}")
                    name = os.path.splitext(file)[0]
                    conn.read_csv(path, target_table=name, quoted=True, encoding='utf-8', auto_drop=True)
                    conn.commit()
                except:
                    print(f"Failed to process {path}")
                    f.write(f"{path}\n")
