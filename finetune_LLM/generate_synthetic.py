"""
generate_synthetic.py — Synthetic Proprietary Telco Training Data Generator
=============================================================================
Uses the Claude API to generate realistic telco Q&A pairs for:
  - Router hardware error codes (Cisco, Nokia, Ericsson)
  - Internal billing codes
  - Proprietary jargon definitions
  - Multi-turn escalation flows

Run: ANTHROPIC_API_KEY=your_key python generate_synthetic.py
Output: synthetic_proprietary.json (ready to concatenate into training data)

Requirements: pip install anthropic
"""

import anthropic
import json
import time
import random
from pathlib import Path

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# ── SEED DATA — Replace / extend with your real codes ────────────────────────
ROUTER_ERRORS = [
    {"code": "E-4531", "hw": "Cisco ASR 9000",    "type": "MPLS label stack overflow"},
    {"code": "E-7712", "hw": "Nokia 7750 SR-12",  "type": "BGP session flap on OSPF adjacency"},
    {"code": "E-2209", "hw": "Ericsson SGSN-MME", "type": "GTP-U tunnel failure — bearer loss"},
    {"code": "E-9901", "hw": "Cisco NCS 5500",    "type": "FIB route table overflow"},
    {"code": "E-3314", "hw": "Juniper MX480",      "type": "Chassis alarm: FPC timeout"},
    {"code": "E-1102", "hw": "Huawei CX600",       "type": "LDP session down — interface flap"},
    {"code": "E-5580", "hw": "Cisco ASR 1000",     "type": "QoS policy-map apply failure"},
    {"code": "E-8823", "hw": "Nokia 7705 SAR",     "type": "RSVP LSP re-route: bandwidth unavailable"},
]

BILLING_CODES = [
    {"code": "RC-4872", "type": "Prorated charge adjustment for mid-cycle plan change"},
    {"code": "RC-1103", "type": "International roaming data surcharge — zone 3"},
    {"code": "RC-2201", "type": "Early termination fee — 24-month contract"},
    {"code": "RC-6614", "type": "Device instalment plan monthly charge"},
    {"code": "RC-3309", "type": "Late payment fee — 15-day overdue"},
    {"code": "RC-7741", "type": "Bundle add-on activation fee"},
    {"code": "RC-0055", "type": "Account reinstatement after suspension"},
    {"code": "RC-9982", "type": "SIM replacement fee — lost or damaged"},
]

JARGON_TERMS = [
    {"term": "MVNO", "full": "Mobile Virtual Network Operator"},
    {"term": "ARPU", "full": "Average Revenue Per User"},
    {"term": "CDR",  "full": "Call Detail Record"},
    {"term": "PCRF", "full": "Policy and Charging Rules Function"},
    {"term": "VoLTE","full": "Voice over Long-Term Evolution"},
    {"term": "IMS",  "full": "IP Multimedia Subsystem"},
    {"term": "HSS",  "full": "Home Subscriber Server"},
    {"term": "PDN-GW","full": "Packet Data Network Gateway"},
    {"term": "eNodeB","full": "Evolved Node B (4G base station)"},
    {"term": "NSSAI", "full": "Network Slice Selection Assistance Information (5G)"},
]

ESCALATION_SCENARIOS = [
    "Customer has been charged for a plan they cancelled 2 months ago",
    "Network outage affecting customer's business — SLA breach possible",
    "Device purchased in-store is faulty — customer wants immediate replacement",
    "Customer received a bill 10x higher than usual after travelling internationally",
]

# ── GENERATORS ────────────────────────────────────────────────────────────────
def gen_router_qa(seed):
    prompt = f"""Generate a realistic telecom technical support Q&A pair.

Error code: {seed['code']}
Hardware: {seed['hw']}
Error type: {seed['type']}

Create a Q&A where a network engineer asks about this error and a senior telco support specialist answers.
The response should include: what the error means, likely cause, diagnostic command, and fix.

Return ONLY valid JSON (no markdown, no backticks):
{{"instruction": "engineer's question about the error", "response": "detailed technical answer (4-6 sentences)", "category": "NETWORK", "intent": "router_error_diagnosis"}}"""
    return prompt

def gen_billing_qa(seed):
    prompt = f"""Generate a realistic telecom billing support Q&A pair.

Billing code: {seed['code']}
Charge type: {seed['type']}

Create a Q&A where a customer asks about a charge on their bill and a billing agent explains it clearly.
Response should be professional, empathetic, and include what the customer can do if they dispute it.

Return ONLY valid JSON (no markdown, no backticks):
{{"instruction": "customer's question about the billing code", "response": "clear billing explanation (3-5 sentences)", "category": "BILLING", "intent": "billing_code_explanation"}}"""
    return prompt

def gen_jargon_qa(seed):
    prompt = f"""Generate a telecom jargon explanation Q&A pair.

Term: {seed['term']} ({seed['full']})

Create a Q&A where a new customer service agent asks what this term means, and a trainer explains it
clearly in plain English with a practical example of when it is used.

Return ONLY valid JSON (no markdown, no backticks):
{{"instruction": "what does {seed['term']} mean in telecom?", "response": "clear jargon explanation with example (3-4 sentences)", "category": "GENERAL", "intent": "jargon_explanation"}}"""
    return prompt

def gen_escalation_flow(scenario):
    prompt = f"""Generate a multi-turn telecom support conversation (escalation flow).

Scenario: {scenario}

Create a 4-turn conversation: customer message → agent response → customer follow-up → agent escalation/resolution.
The agent should be empathetic, professional, and follow escalation best practices.

Return ONLY valid JSON (no markdown, no backticks):
{{"instruction": "Full multi-turn conversation starting with customer complaint: {scenario}", "response": "Turn 1 Agent: [response]\\n\\nTurn 2 Customer: [follow-up]\\n\\nTurn 2 Agent: [escalation or resolution]", "category": "ESCALATION", "intent": "multi_turn_escalation"}}"""
    return prompt

def call_claude(prompt, retries=3):
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )
            text = msg.content[0].text.strip()
            # Strip markdown code blocks if present
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None

# ── MAIN GENERATION LOOP ──────────────────────────────────────────────────────
def main():
    results = []
    total_target = len(ROUTER_ERRORS) + len(BILLING_CODES) + len(JARGON_TERMS) + len(ESCALATION_SCENARIOS)
    print(f"► Generating {total_target} synthetic training examples...")
    print(f"  Router errors:        {len(ROUTER_ERRORS)}")
    print(f"  Billing codes:        {len(BILLING_CODES)}")
    print(f"  Jargon definitions:   {len(JARGON_TERMS)}")
    print(f"  Escalation flows:     {len(ESCALATION_SCENARIOS)}")
    print()

    count = 0
    errors = 0

    # Router error Q&A
    print("► Generating router error Q&A pairs...")
    for seed in ROUTER_ERRORS:
        result = call_claude(gen_router_qa(seed))
        if result:
            results.append(result)
            count += 1
            print(f"  ✓ {seed['code']} on {seed['hw']}")
        else:
            errors += 1
            print(f"  ✗ Failed: {seed['code']}")
        time.sleep(0.5)  # Rate limiting

    # Billing code Q&A
    print("\n► Generating billing code Q&A pairs...")
    for seed in BILLING_CODES:
        result = call_claude(gen_billing_qa(seed))
        if result:
            results.append(result)
            count += 1
            print(f"  ✓ {seed['code']}: {seed['type'][:40]}...")
        else:
            errors += 1
        time.sleep(0.5)

    # Jargon definitions
    print("\n► Generating jargon definition pairs...")
    for seed in JARGON_TERMS:
        result = call_claude(gen_jargon_qa(seed))
        if result:
            results.append(result)
            count += 1
            print(f"  ✓ {seed['term']} ({seed['full']})")
        else:
            errors += 1
        time.sleep(0.5)

    # Escalation flows
    print("\n► Generating multi-turn escalation flows...")
    for scenario in ESCALATION_SCENARIOS:
        result = call_claude(gen_escalation_flow(scenario))
        if result:
            results.append(result)
            count += 1
            print(f"  ✓ Escalation: {scenario[:50]}...")
        else:
            errors += 1
        time.sleep(0.5)

    # Save results
    output_path = Path("synthetic_proprietary.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"✓ Generated {count} examples ({errors} failed)")
    print(f"  Saved to: {output_path}")
    print(f"\nTo add to training pipeline:")
    print(f"  from datasets import Dataset")
    print(f"  import json")
    print(f"  synthetic = Dataset.from_list(json.load(open('synthetic_proprietary.json')))")
    print(f"  combined = concatenate_datasets([your_existing_data, synthetic])")

if __name__ == "__main__":
    main()
