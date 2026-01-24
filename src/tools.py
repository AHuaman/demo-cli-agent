from datetime import datetime
import mlflow
from typing import List, Optional
from sqlmodel import select, col
from src.database import get_session
from src.models import Appointment
from src.settings import settings

# Set tracking URI for MLflow
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

@mlflow.trace
def list_appointments(limit: int = 10, offset: int = 0) -> List[Appointment]:
    """
    List appointments with pagination.
    
    Args:
        limit: Maximum number of appointments to return.
        offset: Number of appointments to skip.
        
    Returns:
        List of Appointment objects.
    """
    with get_session() as session:
        statement = select(Appointment).offset(offset).limit(limit)
        results = session.exec(statement)
        return list(results.all())

@mlflow.trace
def search_appointments(query: str) -> List[Appointment]:
    """
    Search appointments by title, location or description.
    
    Args:
        query: Search string to match against title, location or description.
        
    Returns:
        List of matching Appointment objects.
    """
    with get_session() as session:
        statement = select(Appointment).where(
            (col(Appointment.title).contains(query)) |
            (col(Appointment.description).contains(query)) |
            (col(Appointment.location).contains(query))
        )
        results = session.exec(statement)
        return list(results.all())

@mlflow.trace
def get_appointment(id: int) -> Optional[Appointment]:
    """
    Get a specific appointment by ID.
    
    Args:
        id: The ID of the appointment to retrieve.
        
    Returns:
        The Appointment object if found, None otherwise.
    """
    with get_session() as session:
        return session.get(Appointment, id)

@mlflow.trace
def create_appointment(
    title: str,
    start_time: datetime,
    end_time: datetime,
    location: Optional[str] = None,
    description: Optional[str] = None
) -> Appointment:
    """
    Create a new appointment.
    
    Args:
        title: Title of the appointment.
        start_time: Start time.
        end_time: End time.
        location: Optional location.
        description: Optional description.
        
    Returns:
        The newly created Appointment object.
    """
    with get_session() as session:
        appointment = Appointment(
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=location,
            description=description
        )
        session.add(appointment)
        session.commit()
        session.refresh(appointment)
        return appointment

@mlflow.trace
def delete_appointment(appointment_id: int) -> bool:
    """
    Delete an appointment by ID.
    
    Args:
        appointment_id: The ID of the appointment to delete.
        
    Returns:
        True if deleted, False if not found.
    """
    with get_session() as session:
        appointment = session.get(Appointment, appointment_id)
        if not appointment:
            return False
        session.delete(appointment)
        session.commit()
        return True
