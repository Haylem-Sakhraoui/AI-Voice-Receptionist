"""
Mock domain tools the agent can call mid-conversation.
Replace the mock logic with real calendar/pricing API calls later.
"""
from langchain.tools import tool


@tool
def check_availability(day: str) -> str:
    """Check appointment availability for a given day (e.g. 'Tuesday')."""
    # Mocked - swap for a real calendar API call
    fake_slots = {
        "Monday": ["10:00 AM", "2:00 PM"],
        "Tuesday": ["9:00 AM", "1:00 PM", "4:00 PM"],
        "Wednesday": [],
    }
    slots = fake_slots.get(day, ["11:00 AM", "3:00 PM"])
    if not slots:
        return f"No availability on {day}. Try another day."
    return f"Available slots on {day}: {', '.join(slots)}"


@tool
def get_pricing_estimate(service: str) -> str:
    """Get a rough price estimate for a service (e.g. 'AC repair', 'system replacement')."""
    # Mocked - swap for a real pricing lookup
    pricing = {
        "ac repair": "$150-$450 depending on the issue",
        "system replacement": "$5,000-$12,000 depending on size and SEER rating",
        "maintenance": "$199/year for the bi-annual plan",
    }
    key = service.lower().strip()
    for k, v in pricing.items():
        if k in key:
            return f"Estimated cost for {service}: {v}"
    return f"No estimate on file for '{service}' - a technician can provide an exact quote."


ALL_TOOLS = [check_availability, get_pricing_estimate]
