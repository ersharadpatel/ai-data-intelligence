# Import the database engine
from ai_data_intelligence.database import engine

# Import the database model base
from ai_data_intelligence.database_models import Base

# Import models to register their tables with SQLAlchemy
from ai_data_intelligence.database_models import PredictionHistory

from ai_data_intelligence.database_models import DatasetQualityReport

# Create all database tables
def init_db():

    # Create tables that do not already exist
    Base.metadata.create_all(bind=engine)

    # Confirm database initialization
    print("Database tables created successfully!")


# Run database initialization when executed directly
if __name__ == "__main__":
    init_db()