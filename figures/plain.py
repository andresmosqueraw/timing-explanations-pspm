"""Plain-language readings of the features the paper's figures show, for the
process-owner-facing figures (three-level card, web mock-up).

LABEL reads a feature *at the value it takes on the illustrated case*
(bar labels); NOUN is a noun phrase for use inside a sentence.
"""

LABEL = {
    "relative_position": "the case is near its end", "reliability": "prediction confidence", "deviation": "predicted deviation",
    "available_resources": "only one staff member is free", "Proba_if_Treated": "outcome if we call",
    "Proba_if_Untreated": "outcome if we don't call", "CreditScore_mean": "no credit score yet",
    "CreditScore_max": "no credit score (max)", "CreditScore_std": "no credit score (spread)",
    "CreditScore_sum": "no credit score (sum)", "NumberOfOffers": "only one offer made", "Activity": "still handling leads",
    "timesincelastevent_mean": "time since last event", "timesincecasestart_max": "time since start",
    "open_cases_min": "open cases", "AMOUNT_REQ": "requested amount", "month_min": "month of application",
    "MonthlyCost_max": "a monthly cost of 183", "ApplicationType": "it is a new credit", "FirstWithdrawalAmount_max": "first withdrawal of 6,000",
    "event_nr_mean": "events so far", "open_cases_max": "many cases open at once", "month_mean": "month of application",
}

NOUN = {
    "relative_position": "the case being near its end", "reliability": "the prediction's confidence", "deviation": "the predicted deviation",
    "available_resources": "a free staff member", "Proba_if_Treated": "the expected effect of calling",
    "Proba_if_Untreated": "the expected outcome if nobody calls", "CreditScore_mean": "no credit score has been recorded",
    "CreditScore_max": "the missing credit score", "NumberOfOffers": "the single offer", "Activity": "the last activity",
    "MonthlyCost_max": "the monthly cost (183)", "ApplicationType": "the application type (new credit)",
    "FirstWithdrawalAmount_max": "the first withdrawal amount", "event_nr_mean": "the number of events so far",
    "open_cases_max": "the number of open cases", "AMOUNT_REQ": "the requested amount", "month_mean": "the application month",
}


def label(name):
    return LABEL.get(name, name.replace("_", " "))


def noun(name):
    return NOUN.get(name, name.replace("_", " "))
