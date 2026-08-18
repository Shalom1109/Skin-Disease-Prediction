import sqlite3
import os

def main():
    db_path = 'skinsense.db'
    if not os.path.exists(db_path):
        db_path = os.path.join('backend', 'skinsense.db')
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found. Start the backend first.")
        return

    conn = sqlite3.connect(db_path)
    print("\n" + "="*30)
    print(" SKIN SENSE AI - SQL CONSOLE")
    print("="*30)
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input('sql> ').strip()
            if not query: continue
            if query.lower() == 'exit': break
            
            cur = conn.cursor()
            cur.execute(query)
            
            if query.lower().startswith('select'):
                rows = cur.fetchall()
                if not rows:
                    print("No results found.")
                else:
                    # Print Header if possible
                    colnames = [desc[0] for desc in cur.description]
                    print(" | ".join(colnames))
                    print("-" * (len(" | ".join(colnames)) + 4))
                    for row in rows:
                        print(" | ".join(str(val) for val in row))
            else:
                conn.commit()
                print(f"Success: {cur.rowcount} row(s) affected.")
                
        except sqlite3.Error as e:
            print(f"SQL Error: {e}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            
    conn.close()
    print("\nConsole closed.")

if __name__ == '__main__':
    main()
