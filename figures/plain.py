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
    return "the case has just started" if x < 0.3 else ("the case is near its end" if x > 0.7 else "the case is midway")


def _offers(v):
    x = _num(v)
    if x is None:
        return "offers made"
    return "only one offer made" if int(round(x)) == 1 else f"{int(round(x))} offers made"


LABEL = {
    "relative_position": _position,
    "reliability": lambda v: "prediction confidence",
    "deviation": lambda v: "predicted to end well" if _num(v) == 1 else "predicted to end badly",
    "available_resources": _resources,
    "Proba_if_Treated": lambda v: "outcome if we call",
    "Proba_if_Untreated": lambda v: "outcome if we don't call",
    "CreditScore_mean": _credit("mean"), "CreditScore_max": _credit("max"), "CreditScore_std": _credit("spread"),
    "CreditScore_sum": _credit("sum"), "CreditScore_min": _credit("min"),
    "NumberOfOffers": _offers,
    "Activity": lambda v: f"last activity: {v}" if isinstance(v, str) else "last activity",
    "ApplicationType": lambda v: f"it is a {str(v).lower()}" if isinstance(v, str) else "application type",
    "LoanGoal": lambda v: f"loan goal: {str(v).lower()}" if isinstance(v, str) else "loan goal",
    "MonthlyCost_max": lambda v: f"a monthly cost of {_amount(_num(v))}" if _num(v) else "no monthly cost yet",
    "MonthlyCost_mean": lambda v: f"a mean monthly cost of {_amount(_num(v))}" if _num(v) else "no monthly cost yet",
    "FirstWithdrawalAmount_max": lambda v: f"first withdrawal of {_amount(_num(v))}" if _num(v) else "no withdrawal amount yet",
    "OfferedAmount_mean": lambda v: f"offered amount {_amount(_num(v))}" if _num(v) else "no offer amount yet",
    "RequestedAmount": lambda v: f"requested {_amount(_num(v))}" if _num(v) else "requested amount",
    "AMOUNT_REQ": lambda v: f"requested {_amount(_num(v))}" if _num(v) else "requested amount",
    "event_nr_mean": lambda v: "events so far", "event_nr_max": lambda v: "events so far",
    "open_cases_max": lambda v: "many cases open at once", "open_cases_min": lambda v: "open cases",
    "timesincelastevent_mean": lambda v: "time since last event", "timesincelastevent_max": lambda v: "time since last event",
    "timesincecasestart_max": lambda v: "time since start",
    "month_min": lambda v: "month of application", "month_mean": lambda v: "month of application", "month_max": lambda v: "month of application",
}

NOUN = {
    "relative_position": "the case's progress", "reliability": "the prediction's confidence", "deviation": "the predicted deviation",
    "available_resources": "the free staff", "Proba_if_Treated": "the expected effect of calling",
    "Proba_if_Untreated": "the expected outcome if nobody calls", "CreditScore_mean": "the credit score",
    "CreditScore_max": "the credit score", "NumberOfOffers": "the number of offers", "Activity": "the last activity",
    "MonthlyCost_max": "the monthly cost", "ApplicationType": "the application type",
    "FirstWithdrawalAmount_max": "the first withdrawal amount", "event_nr_mean": "the number of events so far",
    "open_cases_max": "the number of open cases", "AMOUNT_REQ": "the requested amount", "month_mean": "the application month",
}


def label(name, value=None):
    f = LABEL.get(name)
    return f(value) if f else name.replace("_", " ")


def noun(name):
    return NOUN.get(name, name.replace("_", " "))
