"""Plain-language readings of the features the paper's figures show, for the
process-owner-facing figures (three-level card, web mock-up, flow figure).

``label(name, value)`` reads a feature *at the value it takes on the
illustrated case* (bar labels); ``noun(name)`` is a noun phrase for use inside
a sentence. Both fall back to the technical name.
"""


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _amount(v):
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}"


def _credit(agg):
    def f(v):
        x = _num(v)
        if x is None:
            return "credit score"
        return f"no credit score yet ({agg})" if x == 0 else f"credit score {_amount(x)} ({agg})"
    return f


def _resources(v):
    x = _num(v)
    if x is None:
        return "free staff"
    n = int(round(x))
    return {0: "no staff member is free", 1: "only one staff member is free"}.get(n, f"{n} staff members are free")


def _position(v):
    x = _num(v)
    if x is None:
        return "progress of the case"
    # relative_position is prefix_nr over a fixed horizon (a typical case length), not the case's own length
    return "the case is still early" if x < 0.3 else ("the case is running long" if x > 0.7 else "the case is midway")


def _offers(v):
    x = _num(v)
    if x is None:
        return "offers made"
    return "only one offer made" if int(round(x)) == 1 else f"{int(round(x))} offers made"


# Lecturas llanas de los nombres de actividad de los logs BPIC, para que quien no
# conozca el log pueda leer la traza y la etiqueta de "pasos dados hasta ahora".
ACTIVITY = {
    # BPIC2017
    "A_Create Application": "application created",
    "A_Submitted": "submitted by the customer",
    "A_Concept": "first review",
    "A_Accepted": "application accepted",
    "A_Complete": "application completed",
    "A_Validating": "being validated",
    "A_Incomplete": "papers missing",
    "A_Denied": "application denied",
    "A_Cancelled": "application cancelled",
    "A_Pending": "awaiting the customer",
    "W_Handle leads": "staff follows up the lead",
    "W_Complete application": "staff completes the application",
    "W_Call after offers": "staff calls about the offer",
    "W_Validate application": "staff checks the application",
    "W_Call incomplete files": "staff calls about missing papers",
    "W_Assess potential fraud": "fraud check",
    "W_Shortened completion": "shortened completion",
    "O_Create Offer": "offer prepared",
    "O_Created": "offer created",
    "O_Sent (mail and online)": "offer sent",
    "O_Sent (online only)": "offer sent online",
    "O_Returned": "offer returned",
    "O_Accepted": "offer accepted",
    "O_Refused": "offer refused",
    "O_Cancelled": "offer cancelled",
    # BPIC2012
    "A_SUBMITTED": "submitted by the customer",
    "A_PARTLYSUBMITTED": "partly submitted",
    "A_PREACCEPTED": "pre-accepted",
    "A_ACCEPTED": "application accepted",
    "A_FINALIZED": "application finalised",
    "A_REGISTERED": "application registered",
    "A_APPROVED": "application approved",
    "A_ACTIVATED": "loan activated",
    "A_DECLINED": "application declined",
    "A_CANCELLED": "application cancelled",
    "O_SELECTED": "offer selected",
    "O_CREATED": "offer created",
    "O_SENT": "offer sent",
    "O_SENT_BACK": "offer returned",
    "O_ACCEPTED": "offer accepted",
    "O_DECLINED": "offer declined",
    "O_CANCELLED": "offer cancelled",
    "W_Completeren aanvraag": "staff completes the application",
    "W_Nabellen offertes": "staff calls about the offer",
    "W_Valideren aanvraag": "staff checks the application",
    "W_Nabellen incomplete dossiers": "staff calls about missing papers",
    "W_Afhandelen leads": "staff follows up the lead",
    "W_Beoordelen fraude": "fraud check",
}


def activity(name):
    """Lectura llana de una actividad; si no está, se le quita la etiqueta A_/O_/W_."""
    if not isinstance(name, str):
        return "a step of the case"
    if name in ACTIVITY:
        return ACTIVITY[name]
    stem = name[2:] if name[:2] in ("A_", "O_", "W_") else name
    return stem.replace("_", " ").lower()


LABEL = {
    "relative_position": _position,
    "reliability": lambda v: "prediction confidence",
    "deviation": lambda v: "predicted to end well" if _num(v) == 1 else "predicted to end badly",
    "available_resources": _resources,
    "Proba_if_Treated": lambda v: "if a further offer is made",
    "Proba_if_Untreated": lambda v: "if no further offer is made",
    "CreditScore_mean": _credit("mean"), "CreditScore_max": _credit("max"), "CreditScore_std": _credit("spread"),
    "CreditScore_sum": _credit("sum"), "CreditScore_min": _credit("min"),
    "NumberOfOffers": _offers,
    # the encoding is the running lexicographic maximum of the activities so far, not the last one.
    # No se nombra la actividad: la traza de pasos ya va arriba en la misma figura.
    "Activity": lambda v: "the steps taken so far",
    "NumberOfTerms_sum": lambda v: "terms offered so far", "NumberOfTerms_max": lambda v: "longest term offered",
    "org:resource": lambda v: "the employee handling it", "open_cases_mean": lambda v: "other cases open now",
    "ApplicationType": lambda v: f"a {str(v).lower()} application" if isinstance(v, str) else "application type",
    "LoanGoal": lambda v: f"goal: {str(v).lower()}" if isinstance(v, str) else "loan goal",
    "OfferedAmount_std": lambda v: "how much offers vary", "OfferedAmount_max": lambda v: "largest amount offered",
    "OfferedAmount_min": lambda v: "smallest amount offered", "OfferedAmount_sum": lambda v: "total amount offered",
    "MonthlyCost_max": lambda v: f"a monthly cost of {_amount(_num(v))}" if _num(v) else "no monthly cost yet",
    "MonthlyCost_mean": lambda v: f"a mean monthly cost of {_amount(_num(v))}" if _num(v) else "no monthly cost yet",
    "FirstWithdrawalAmount_max": lambda v: f"first withdrawal of {_amount(_num(v))}" if _num(v) else "no withdrawal amount yet",
    "OfferedAmount_mean": lambda v: f"offered amount {_amount(_num(v))}" if _num(v) else "no offer amount yet",
    "RequestedAmount": lambda v: f"loan requested: {_amount(_num(v))}" if _num(v) else "requested amount",
    "AMOUNT_REQ": lambda v: f"loan requested: {_amount(_num(v))}" if _num(v) else "requested amount",
    "event_nr_mean": lambda v: "steps so far", "event_nr_max": lambda v: "steps so far",
    "open_cases_max": lambda v: "other cases open now", "open_cases_min": lambda v: "other cases open now",
    "timesincelastevent_mean": lambda v: "time since last event", "timesincelastevent_max": lambda v: "time since last event",
    "timesincecasestart_max": lambda v: "time since start",
    "month_min": lambda v: "month of application", "month_mean": lambda v: "month of application", "month_max": lambda v: "month of application",
}

NOUN = {
    "relative_position": "the case's progress", "reliability": "the prediction's confidence", "deviation": "the predicted deviation",
    "available_resources": "the free staff", "Proba_if_Treated": "the expected outcome with a further offer",
    "Proba_if_Untreated": "the expected outcome without a further offer", "CreditScore_mean": "the credit score",
    "CreditScore_max": "the credit score", "Activity": "the activities so far",
    "MonthlyCost_max": "the monthly cost", "ApplicationType": "the application type",
    "FirstWithdrawalAmount_max": "the first withdrawal amount", "event_nr_mean": "the number of events so far",
    "open_cases_max": "the number of open cases", "AMOUNT_REQ": "the requested amount", "month_mean": "the application month",
}


def label(name, value=None):
    f = LABEL.get(name)
    return f(value) if f else name.replace("_", " ")


def noun(name):
    return NOUN.get(name, name.replace("_", " "))
