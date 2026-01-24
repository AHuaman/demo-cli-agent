import pytest
import mlflow
import tempfile
import shutil
from sqlmodel import create_engine, SQLModel, Session
from datetime import datetime, timedelta
from src.models import Appointment
from src import tools
from src.database import get_session
import contextlib

@pytest.fixture(scope="module", autouse=True)
def setup_mlflow():
    # Create a temp directory for MLflow tracking
    temp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file://{temp_dir}")
    yield
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine

@pytest.fixture(name="session_factory")
def session_factory_fixture(engine):
    # Override the get_session dependency in tools to use our test engine
    @contextlib.contextmanager
    def get_session_override():
        with Session(engine) as session:
            yield session
    
    # Monkeypatch src.tools.get_session
    old_get_session = tools.get_session
    tools.get_session = get_session_override
    yield
    tools.get_session = old_get_session

@pytest.fixture
def sample_appointments(engine):
    with Session(engine) as session:
        app1 = Appointment(
            title="Team Meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            location="Room A",
            description="Weekly sync"
        )
        app2 = Appointment(
            title="Doctor Appointment",
            start_time=datetime.now() + timedelta(days=1),
            end_time=datetime.now() + timedelta(days=1, hours=1),
            location="Clinic",
            description="Checkup"
        )
        app3 = Appointment(
            title="Lunch with Client",
            start_time=datetime.now() + timedelta(days=2),
            end_time=datetime.now() + timedelta(days=2, hours=2),
            location="Restaurant",
            description="Project discussion"
        )
        session.add(app1)
        session.add(app2)
        session.add(app3)
        session.commit()
        return [app1, app2, app3]

def test_list_appointments(session_factory, sample_appointments):
    results = tools.list_appointments()
    assert len(results) == 3
    
    # Test pagination
    results = tools.list_appointments(limit=1, offset=0)
    assert len(results) == 1
    assert results[0].title == "Team Meeting"
    
    results = tools.list_appointments(limit=1, offset=1)
    assert len(results) == 1
    assert results[0].title == "Doctor Appointment"

def test_search_appointments(session_factory, sample_appointments):
    # Search by title
    results = tools.search_appointments("Doctor")
    assert len(results) == 1
    assert results[0].title == "Doctor Appointment"
    
    # Search by location
    results = tools.search_appointments("Room A")
    assert len(results) == 1
    assert results[0].title == "Team Meeting"
    
    # Search by description
    results = tools.search_appointments("Project")
    assert len(results) == 1
    assert results[0].title == "Lunch with Client"
    
    # No match
    results = tools.search_appointments("Nonexistent")
    assert len(results) == 0

def test_get_appointment(session_factory, sample_appointments):
    # Get existing
    app = tools.get_appointment(1)
    assert app is not None
    assert app.title == "Team Meeting"
    
    # Get non-existent
    app = tools.get_appointment(999)
    assert app is None
