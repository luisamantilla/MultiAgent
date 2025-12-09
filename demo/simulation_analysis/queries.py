import duckdb

def calculate_cell_count(df):
    return duckdb.query("""
                        SELECT time, COUNT(*) as cell_count 
                        FROM df 
                        GROUP BY time 
                        ORDER BY time
                        """
                    ).to_df().to_dict(orient="records")

def calculate_pdl1_state_count(df):
    return duckdb.query("""
                        SELECT 
                            time, 
                            cell_state, 
                            COUNT(*) as cell_count,
                            COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY time) as ratio
                        FROM df 
                        GROUP BY time, cell_state 
                        ORDER BY time
                        """
                    ).to_df().to_dict(orient="records")
    
def calculate_cell_division(df):
    return duckdb.query("""
                 SELECT birth_time, COUNT(*) as number_of_births
                 FROM (
                     SELECT cell, MIN(time) as birth_time
                     FROM df
                     GROUP BY cell
                 )
                 GROUP BY birth_time
                 ORDER BY birth_time 
                 """
            ).to_df().to_dict(orient="records")
    
def calculate_pd1_plus_ifng(df):
    return duckdb.query("""
                    SELECT time, AVG(IFNg)
                    FROM df
                    WHERE cell_state = 'PD1p'
                    GROUP BY time
                    ORDER BY time
                 """).to_df().to_dict(orient="records")
    
def calculate_pd1_plus_average_movement(df):
    return duckdb.query("""
                        SELECT time, AVG(dx + dy)
                        FROM (
                            SELECT
                                time,
                                cell,
                                X - LAG(X) OVER (PARTITION BY cell ORDER BY time) as dx,
                                Y - LAG(Y) OVER (PARTITION BY cell ORDER BY time) as dy
                            FROM df
                            WHERE cell_state = 'PD1p'
                        )
                        WHERE dx IS NOT NULL AND dy IS NOT NULL
                        GROUP BY time
                        ORDER BY time
                        """).to_df().to_dict(orient="records")


if __name__ == "__main__":
    import pandas as pd

    # Update paths to match your setup
    df = pd.read_csv("demo/simulation_analysis/tcell_plot.csv")
    tumor_df = pd.read_csv("demo/simulation_analysis/tumor_plot.csv")

    print("📊 PD1+ IFNg levels (first 5):")
    print(calculate_pd1_plus_ifng(df)[:5])

    print("\n🏃 PD1+ average movement (first 5):")
    print(calculate_pd1_plus_average_movement(df)[:5])

    print("\n🧫 Cell count (first 5):")
    print(calculate_cell_count(df)[:5])
