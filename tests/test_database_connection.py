# Import the database engine
from ai_data_intelligence.database import engine

# Test the database connection
def test_database_connection():

    # Establish a connection with the database
    with engine.connect() as connection:

        # Execute a simple SQL query
        result = connection.exec_driver_sql("SELECT 1")

        # Verify that the database responds correctly
        assert result.scalar() == 1