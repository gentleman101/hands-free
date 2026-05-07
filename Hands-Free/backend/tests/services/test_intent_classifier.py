"""Intent classifier tests."""
from __future__ import annotations

from backend.services.intent_classifier import classify


def test_billing_en():
    assert classify("what is my electricity bill this month") == "billing"


def test_billing_ar():
    assert classify("كم فاتورتي") == "billing"


def test_location_en():
    assert classify("where is the nearest happiness center") == "location"


def test_location_ar():
    assert classify("أين مركز السعادة") == "location"


def test_outage_en():
    assert classify("there is a power outage in my area") == "outage"


def test_outage_ar():
    assert classify("انقطاع الكهرباء عندي") == "outage"


def test_account_en():
    assert classify("I want to update my account details") == "account"


def test_service_request_en():
    assert classify("I need a new electricity connection") == "service_request"


def test_empty_falls_back_to_general():
    assert classify("") == "general"


def test_low_confidence_falls_back_to_general():
    assert classify("hello good morning") == "general"
