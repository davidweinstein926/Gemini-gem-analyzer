import pandas as pd


def check_or_delete_gems():
    print("\U0001F9ED What would you like to do?")
    print("   C - Check if a gem is in the database")
    print("   D - Delete a gem from the database")
    print("   L - List C-series gems and their light sources")
    print("   Q - Quit without changes")

    while True:
        mode = input("➡️ Enter your choice (C/D/L/Q): ").strip().upper()
        if mode in {'C', 'D'}:
            check_only = (mode == 'C')
            break
        elif mode == 'L':
            try:
                db = pd.read_csv('gemini_db_long.csv', dtype={'gem_id': str, 'full_name': str})
                db['gem_id'] = db['gem_id'].astype(str).str.strip().str.upper()
                print("\n📋 C-Series Gem Summary:")
                list_mode = input("🔍 Enter A for All or R for Range (A/R): ").strip().upper()
                if list_mode == 'A':
                    c_gems = db[db['gem_id'].str.startswith('C')]
                elif list_mode == 'R':
                    start = input("🔢 From (e.g. C1000): ").strip().upper()
                    end = input("🔢 To (e.g. C1020): ").strip().upper()
                    db['numeric_id'] = pd.to_numeric(db['gem_id'].str.extract(r'^C(\d{4})')[0], errors='coerce')
                    start_num = int(start[1:])
                    end_num = int(end[1:])
                    c_gems = db[(db['gem_id'].str.startswith('C')) &
                                (db['numeric_id'] >= start_num) &
                                (db['numeric_id'] <= end_num)]
                else:
                    print("⚠️ Invalid option. Returning to main menu.")
                    return

                if c_gems.empty:
                    print("⚠️ No C-series gems found in that range.")
                    return

                summary = c_gems.groupby('gem_id')['light_source'].unique().reset_index()
                for _, row in summary.iterrows():
                    sources = ','.join(sorted(row['light_source']))
                    print(f"{row['gem_id']}: {sources}")
            except Exception as e:
                print(f"❌ Failed to load or summarize database: {e}")
            return

        elif mode == 'Q':
            print("\U0001F44B Exiting without changes.")
            return
        else:
            print("⚠️ Invalid choice. Please enter C, D, L, or Q.")

    db_file = 'gemini_db_long.csv'
    try:
        db = pd.read_csv(db_file, dtype={'gem_id': str, 'full_name': str})
    except FileNotFoundError:
        print("❌ Database file 'gemini_db_long.csv' not found.")
        return

    while True:
        action = "check" if check_only else "delete"
        gem_to_query = input(f"\U0001F539 Enter a gem ID or full name to {action} (e.g. 638470141 or C1001Bc1): ").strip()

        try:
            gem_id_candidate = int(gem_to_query)
        except ValueError:
            gem_id_candidate = None

        query = gem_to_query.strip().lower()
        matches = db[
            (db['gem_id'].astype(str).str.lower() == query) |
            (db['full_name'].astype(str).str.lower() == query) |
            (db['full_name'].astype(str).str.lower().str.startswith(query))
        ]

        if not matches.empty:
            print("\n\U0001F4CB Matching entries:")
            print(matches[['gem_id', 'light_source', 'full_name', 'source_file']])
            if check_only:
                print(f"✅ Gem {gem_to_query} exists in the database with the following entries:")
                print(matches[['gem_id', 'light_source', 'full_name', 'source_file']].to_string(index=False))
            else:
                db = db[~((db['gem_id'].astype(str).str.lower() == query) | (db['full_name'].astype(str).str.lower() == query))]
                print(f"✅ Removed all entries for: {gem_to_query}")
        else:
            print(f"⚠️ No entries found for: {gem_to_query}")

        prompt = "➕ Check another gem? (y/n): " if check_only else "➕ Delete another gem? (y/n): "
        cont = input(prompt).strip().lower()
        if cont != 'y':
            break

    save = input("💾 Do you want to save the current database to 'gemini_db_long.csv'? (y/n): ").strip().lower()
    if save == 'y':
        db.to_csv(db_file, index=False)
        print("💾 Database saved successfully.")
    else:
        print("❎ Changes were not saved.")


if __name__ == "__main__":
    check_or_delete_gems()
