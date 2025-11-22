import sqlite3
import pandas as pd

# Set display options to show all columns cleanly
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def show_tables():
    try:
        # Connect to the database
        conn = sqlite3.connect("app_data.db")
        
        print("-" * 50)
        print("📢 TABLE: USERS (Username | Hashed Password)")
        print("-" * 50)
        try:
            # Read SQL directly into a pandas DataFrame for pretty printing
            users = pd.read_sql_query("SELECT * FROM users", conn)
            if users.empty:
                print("[Empty Table]")
            else:
                print(users)
        except Exception as e:
            print(f"Error reading users: {e}")

        print("\n" + "-" * 50)
        print("📝 TABLE: FEEDBACK (Logs & Comments)")
        print("-" * 50)
        try:
            feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
            if feedback.empty:
                print("[Empty Table]")
            else:
                print(feedback)
        except Exception as e:
            print(f"Error reading feedback: {e}")
            
        conn.close()
        
    except Exception as e:
        print(f"Could not connect to database: {e}")

if __name__ == "__main__":
    show_tables()