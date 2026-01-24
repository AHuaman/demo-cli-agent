import pytest
import mlflow
import tempfile
import shutil
from sqlmodel import create_engine, SQLModel, Session, select
from datetime import datetime, timedelta
from src.models import Appointment
from src import tools
import contextlib

@pytest.fixture(scope="module", autouse=True)
def setup_mlflow():
    temp_dir = tempfile.mkdtemp()
    db_path = f"{temp_dir}/mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    shutil.rmtree(temp_dir)

@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine

@pytest.fixture(name="session_factory", autouse=True)
def session_factory_fixture(engine, monkeypatch):
    @contextlib.contextmanager
    def get_session_override():
        with Session(engine) as session:
            yield session
    monkeypatch.setattr("src.tools.get_session", get_session_override)

def test_create_appointment(engine):
    now = datetime.now()
    app = tools.create_appointment(
        title="New Appointment",
        start_time=now,
        end_time=now + timedelta(hours=1),
        location="Office",
        description="Test desc"
    )
    
    assert app.id is not None
    assert app.title == "New Appointment"
    
    with Session(engine) as session:
        db_app = session.get(Appointment, app.id)
        assert db_app is not None
        assert db_app.title == "New Appointment"

def test_delete_appointment(engine):
    # First create one
    now = datetime.now()
    with Session(engine) as session:
        app = Appointment(
            title="To Delete",
            start_time=now,
            end_time=now + timedelta(hours=1)
        )
        session.add(app)
        session.commit()
        app_id = app.id

    # Delete it
    success = tools.delete_appointment(app_id)
    assert success is True
    
    # Verify it's gone
    with Session(engine) as session:
        db_app = session.get(Appointment, app_id)
        assert db_app is None

def test_delete_nonexistent_appointment():
    success = tools.delete_appointment(999)
    assert success is False
