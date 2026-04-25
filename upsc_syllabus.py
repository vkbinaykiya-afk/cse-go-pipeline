"""
upsc_syllabus.py — UPSC CSE syllabus structured for topic planning.

Source: Official UPSC syllabus, Paper I (Prelims) + GS Papers II, III, IV (Mains).
Used by daily_pipeline.py planner to ensure topic diversity and syllabus alignment.
"""

# ── PRELIMS PAPER I ───────────────────────────────────────────────────────────

PRELIMS = {
    "Current Affairs": [
        "Current events of national and international importance",
    ],
    "History": [
        "History of India and Indian National Movement",
    ],
    "Geography": [
        "Physical, Social, Economic Geography of India and the World",
    ],
    "Polity": [
        "Constitution, Political System, Panchayati Raj, Public Policy, Rights Issues",
    ],
    "Economy": [
        "Sustainable Development, Poverty, Inclusion, Demographics, Social Sector initiatives",
    ],
    "Environment": [
        "Environmental Ecology, Bio-diversity and Climate Change",
    ],
    "Science & Technology": [
        "General Science",
    ],
}

# ── MAINS GS PAPER II (Governance, Polity, IR) ───────────────────────────────

MAINS_GS2 = {
    "Polity & Constitution": [
        "Indian Constitution — historical underpinnings, evolution, features, amendments, significant provisions, basic structure",
        "Functions and responsibilities of Union and States, federal structure, devolution of powers",
        "Separation of powers, dispute redressal mechanisms and institutions",
        "Comparison of Indian constitutional scheme with other countries",
        "Parliament and State Legislatures — structure, functioning, powers & privileges",
        "Executive and Judiciary — structure, organization, functioning; Ministries and Departments",
        "Salient features of the Representation of People's Act",
        "Constitutional posts, powers, functions of various Constitutional Bodies",
        "Statutory, regulatory and quasi-judicial bodies",
    ],
    "Governance": [
        "Government policies and interventions for development in various sectors",
        "Development processes — role of NGOs, SHGs, various groups and associations",
        "Welfare schemes for vulnerable sections; mechanisms, laws, institutions for protection",
        "Issues relating to Social Sector: Health, Education, Human Resources",
        "Issues relating to poverty and hunger",
        "Governance, transparency, accountability, e-governance — applications, models, successes",
        "Citizens charters, transparency & accountability",
        "Role of civil services in a democracy",
    ],
    "International Relations": [
        "India and its neighborhood — relations",
        "Bilateral, regional and global groupings and agreements involving India",
        "Effect of policies of developed and developing countries on India's interests",
        "Indian diaspora",
        "Important international institutions, agencies and fora — structure, mandate",
    ],
}

# ── MAINS GS PAPER III (Economy, Environment, Security) ─────────────────────

MAINS_GS3 = {
    "Economy": [
        "Indian Economy — planning, mobilization of resources, growth, development, employment",
        "Inclusive growth and issues arising from it",
        "Government Budgeting",
        "Major crops, cropping patterns, irrigation systems, agricultural marketing",
        "Farm subsidies, MSP, Public Distribution System, food security, buffer stocks",
        "Food processing industries — scope, location, supply chain management",
        "Land reforms in India",
        "Liberalization effects on economy, industrial policy and industrial growth",
        "Infrastructure — Energy, Ports, Roads, Airports, Railways",
        "Investment models",
    ],
    "Science & Technology": [
        "Science and Technology developments and their applications in everyday life",
        "Achievements of Indians in science & technology; indigenization of technology",
        "IT, Space, Computers, robotics, nano-technology, bio-technology, intellectual property rights",
    ],
    "Environment": [
        "Conservation, environmental pollution and degradation, environmental impact assessment",
        "Disaster and disaster management",
    ],
    "Internal Security": [
        "Linkages between development and spread of extremism",
        "Role of external state and non-state actors in internal security challenges",
        "Cyber security, money-laundering, social networking and internal security",
        "Security challenges in border areas; organized crime and terrorism linkages",
        "Various security forces and agencies and their mandate",
    ],
}

# ── MAINS GS PAPER II — History, Culture, Geography, Society ────────────────

MAINS_GS1 = {
    "History & Culture": [
        "Indian culture — Art Forms, Literature and Architecture from ancient to modern times",
        "Modern Indian history from mid-18th century — significant events, personalities, issues",
        "The Freedom Struggle — stages and important contributors from different parts",
        "Post-independence consolidation and reorganization",
        "History of the world — industrial revolution, world wars, colonization, decolonization, political philosophies",
    ],
    "Society": [
        "Salient features of Indian Society, Diversity of India",
        "Role of women and women's organizations, population, poverty, urbanization",
        "Effects of globalization on Indian society",
        "Social empowerment, communalism, regionalism, secularism",
    ],
    "Geography": [
        "Salient features of world's physical geography",
        "Distribution of key natural resources across the world including South Asia",
        "Factors responsible for location of primary, secondary, tertiary sector industries",
        "Important geophysical phenomena — earthquakes, Tsunami, Volcanic activity, cyclone",
        "Geographical features and their location; changes in water-bodies, ice-caps, flora and fauna",
    ],
}

# ── COMBINED TOPIC BANK ───────────────────────────────────────────────────────
# Flat structure for easy sampling by the planner.
# Each entry: (subject_label, topic_string)

ALL_TOPICS: list[tuple[str, str]] = []

for subject, topics in {**PRELIMS, **MAINS_GS1, **MAINS_GS2, **MAINS_GS3}.items():
    for topic in topics:
        ALL_TOPICS.append((subject, topic))


def get_syllabus_text() -> str:
    """Return the full syllabus as a formatted string for use in planner prompts."""
    sections = {
        "PRELIMS Paper I": PRELIMS,
        "MAINS GS-I (History, Culture, Geography, Society)": MAINS_GS1,
        "MAINS GS-II (Governance, Polity, International Relations)": MAINS_GS2,
        "MAINS GS-III (Economy, Environment, Science, Security)": MAINS_GS3,
    }
    lines = []
    for section, subjects in sections.items():
        lines.append(f"\n{section}:")
        for subject, topics in subjects.items():
            lines.append(f"  [{subject}]")
            for t in topics:
                lines.append(f"    - {t}")
    return "\n".join(lines)
