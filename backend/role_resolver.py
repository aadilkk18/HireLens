from sentence_transformers import SentenceTransformer, util

ROLES = [
    # Software Development
    {"title": "Junior Software Engineer", "description": "entry level software development, coding, building applications"},
    {"title": "Backend Developer", "description": "server side development, APIs, databases, backend systems"},
    {"title": "Frontend Developer", "description": "UI development, React, web interfaces, client side code"},
    {"title": "Full Stack Developer", "description": "both frontend and backend web development"},
    {"title": "Mobile App Developer", "description": "Android, iOS, Flutter, React Native mobile applications"},
    {"title": "Game Developer", "description": "video game development, Unity, Unreal Engine, game design"},
    {"title": "WordPress Developer", "description": "WordPress themes, plugins, PHP, CMS development"},
    {"title": "Software Engineer (Mid/Senior)", "description": "experienced software development, system design, architecture"},
    {"title": "Software Architect", "description": "system architecture, technical leadership, large scale design"},
    {"title": "Embedded Systems Engineer", "description": "firmware, microcontrollers, IoT, embedded C programming"},

    # AI / Data
    {"title": "AI/ML Engineer", "description": "machine learning, artificial intelligence, model training, deployment"},
    {"title": "Data Scientist", "description": "statistical modeling, predictive analytics, machine learning research"},
    {"title": "Data Analyst", "description": "analyzing data, dashboards, reports, business intelligence, SQL, Excel"},
    {"title": "Data Engineer", "description": "data pipelines, ETL, data warehousing, big data systems"},
    {"title": "NLP Engineer", "description": "natural language processing, chatbots, text analysis, LLMs"},
    {"title": "Computer Vision Engineer", "description": "image processing, object detection, CNNs, visual AI systems"},

    # Infrastructure / QA
    {"title": "DevOps Engineer", "description": "CI/CD, cloud infrastructure, deployment, servers, automation"},
    {"title": "Cloud Engineer", "description": "AWS, Azure, GCP, cloud architecture and infrastructure"},
    {"title": "QA Engineer / Test Automation", "description": "software testing, quality assurance, test automation, bug tracking"},
    {"title": "Database Administrator", "description": "database management, SQL, data storage systems, backups"},
    {"title": "IT Support / Network Administrator", "description": "IT infrastructure, networking, technical support, system administration"},
    {"title": "Cybersecurity Analyst", "description": "security, penetration testing, vulnerability assessment, information security"},
    {"title": "System Administrator", "description": "server management, Linux, Windows Server, IT operations"},

    # Design
    {"title": "UI/UX Designer", "description": "design, user experience, wireframes, prototyping, Figma"},
    {"title": "Graphic Designer", "description": "visual design, branding, Photoshop, Illustrator, marketing creatives"},
    {"title": "Product Designer", "description": "end to end product design, user research, design systems"},
    {"title": "Motion Graphics / Animator", "description": "animation, motion graphics, video effects, After Effects"},
    {"title": "3D Artist / Animator", "description": "3D modeling, rendering, Blender, Maya, visual effects"},
    {"title": "Video Editor", "description": "video editing, Premiere Pro, post production, content editing"},
    {"title": "Photographer / Videographer", "description": "photography, videography, editing, visual media production"},

    # Business / Management
    {"title": "Product Manager", "description": "product strategy, roadmaps, cross functional coordination, requirements"},
    {"title": "Project Manager", "description": "project planning, timelines, scrum, agile, team coordination"},
    {"title": "Business Analyst", "description": "business requirements, process analysis, stakeholder communication"},
    {"title": "Operations Manager", "description": "business operations, process improvement, logistics"},
    {"title": "Management Trainee", "description": "entry level rotational management program, business fundamentals"},
    {"title": "Administrative Officer", "description": "office administration, coordination, scheduling, records"},
    {"title": "Executive Assistant", "description": "supporting executives, scheduling, correspondence, office management"},
    {"title": "Office Assistant / Clerk", "description": "clerical work, filing, data entry, office support"},

    # HR
    {"title": "HR Executive / Recruiter", "description": "human resources, hiring, talent acquisition, employee relations"},
    {"title": "HR Manager", "description": "HR strategy, policy, employee relations, organizational development"},
    {"title": "Talent Acquisition Specialist", "description": "recruitment, sourcing candidates, interviewing, hiring pipelines"},
    {"title": "Payroll Officer", "description": "payroll processing, salary administration, compliance"},
    {"title": "Training and Development Officer", "description": "employee training, learning programs, skill development"},

    # Marketing / Sales
    {"title": "Digital Marketing Executive", "description": "SEO, social media marketing, Google Ads, campaign management"},
    {"title": "SEO Specialist", "description": "search engine optimization, keyword research, website ranking"},
    {"title": "Content Writer", "description": "writing, copywriting, blogging, content strategy"},
    {"title": "Social Media Manager", "description": "social media strategy, content scheduling, community management"},
    {"title": "Sales Executive", "description": "sales, business development, client acquisition, targets"},
    {"title": "Telesales / Telemarketing Representative", "description": "cold calling, phone sales, lead generation"},
    {"title": "Brand Manager", "description": "brand strategy, marketing campaigns, positioning"},
    {"title": "Marketing Manager", "description": "marketing strategy, campaigns, team leadership, budgets"},
    {"title": "E-commerce Executive", "description": "online store management, Shopify, product listings, order fulfillment"},
    {"title": "Copywriter", "description": "advertising copy, marketing text, brand voice"},
    {"title": "Public Relations Officer", "description": "PR, media relations, press releases, corporate communications"},

    # Finance / Accounting
    {"title": "Accountant / Finance Executive", "description": "accounting, bookkeeping, financial reporting, taxation"},
    {"title": "Financial Analyst", "description": "financial modeling, forecasting, investment analysis"},
    {"title": "Auditor", "description": "internal audit, external audit, compliance, financial review"},
    {"title": "Tax Consultant", "description": "taxation, tax filing, compliance, FBR regulations"},
    {"title": "Bank Officer", "description": "banking operations, customer accounts, branch banking"},
    {"title": "Investment Banking Analyst", "description": "investment banking, mergers, valuations, capital markets"},
    {"title": "Actuary", "description": "risk assessment, insurance calculations, statistical analysis"},
    {"title": "Bookkeeper", "description": "recording transactions, ledgers, basic accounting"},

    # Customer / Support
    {"title": "Customer Support Representative", "description": "customer service, support tickets, client communication"},
    {"title": "Call Center Agent", "description": "inbound/outbound calls, customer queries, BPO"},
    {"title": "Virtual Assistant", "description": "administrative support, scheduling, remote assistance"},
    {"title": "Receptionist / Front Desk", "description": "front desk, greeting visitors, phone handling"},

    # Logistics / Supply Chain
    {"title": "Supply Chain Executive", "description": "supply chain management, procurement, vendor coordination"},
    {"title": "Logistics Coordinator", "description": "shipping, freight, distribution, delivery coordination"},
    {"title": "Warehouse Manager", "description": "warehouse operations, inventory management, stock control"},
    {"title": "Procurement Officer", "description": "purchasing, vendor negotiation, sourcing materials"},
    {"title": "Driver / Delivery Rider", "description": "driving, delivery, transportation, courier services"},
    {"title": "Import/Export Officer", "description": "international trade, customs, shipping documentation"},

    # Retail
    {"title": "Retail Store Manager", "description": "retail operations, store management, sales targets, staff supervision"},
    {"title": "Sales Associate / Cashier", "description": "retail sales, customer assistance, cash handling"},
    {"title": "Merchandiser", "description": "product display, inventory, retail merchandising"},

    # Engineering (non-software)
    {"title": "Electrical Engineer", "description": "electrical systems, circuit design, power engineering"},
    {"title": "Mechanical Engineer", "description": "mechanical design, manufacturing, CAD, product engineering"},
    {"title": "Civil Engineer", "description": "construction, structural design, site engineering"},
    {"title": "Chemical Engineer", "description": "chemical processes, plant operations, process engineering"},
    {"title": "Industrial Engineer", "description": "process optimization, manufacturing efficiency, quality systems"},
    {"title": "Site Engineer / Site Supervisor", "description": "construction site management, on site engineering supervision"},
    {"title": "Architect", "description": "building design, architecture, construction planning"},
    {"title": "Quantity Surveyor", "description": "construction cost estimation, quantity surveying, contracts"},
    {"title": "Textile Engineer", "description": "textile manufacturing, fabric production, quality control"},
    {"title": "Petroleum Engineer", "description": "oil and gas extraction, drilling, reservoir engineering"},

    # Trades
    {"title": "Electrician (Trade)", "description": "electrical wiring, installation, repair, trade work"},
    {"title": "Plumber", "description": "plumbing, pipe fitting, water systems repair"},
    {"title": "Mechanic / Auto Technician", "description": "vehicle repair, automotive maintenance, mechanics"},
    {"title": "Welder / Fabricator", "description": "welding, metal fabrication, industrial trade work"},
    {"title": "HVAC Technician", "description": "heating, ventilation, air conditioning installation and repair"},
    {"title": "Carpenter", "description": "woodworking, furniture making, construction carpentry"},
    {"title": "Machine Operator", "description": "operating industrial or manufacturing machinery"},

    # Healthcare
    {"title": "Doctor / Physician", "description": "medical practice, patient care, diagnosis, treatment"},
    {"title": "Nurse", "description": "nursing, patient care, hospital and clinical support"},
    {"title": "Pharmacist", "description": "pharmacy, medication dispensing, drug counseling"},
    {"title": "Lab Technician", "description": "laboratory testing, sample analysis, diagnostics"},
    {"title": "Medical Officer", "description": "clinical care, hospital medical duties, patient treatment"},
    {"title": "Dentist", "description": "dental care, oral health, dental procedures"},
    {"title": "Physiotherapist", "description": "physical therapy, rehabilitation, patient mobility treatment"},
    {"title": "Radiologist / Imaging Technician", "description": "medical imaging, X-ray, MRI, CT scan operation"},
    {"title": "Healthcare Administrator", "description": "hospital administration, healthcare management, operations"},
    {"title": "Veterinarian", "description": "animal medical care, veterinary treatment"},

    # Hospitality / Culinary
    {"title": "Chef / Cook", "description": "cooking, culinary arts, kitchen management, food preparation, restaurant"},
    {"title": "Restaurant Manager", "description": "restaurant operations, food service management, staff supervision"},
    {"title": "Baker / Pastry Chef", "description": "baking, pastry, desserts, bread making"},
    {"title": "Hotel / Hospitality Staff", "description": "hospitality, front desk, guest services, hotel management"},
    {"title": "Waiter / Waitress / Server", "description": "food service, serving customers, restaurant floor staff"},
    {"title": "Barista", "description": "coffee preparation, cafe service, customer facing food service"},
    {"title": "Event Manager", "description": "event planning, coordination, weddings, corporate events"},
    {"title": "Travel Consultant", "description": "travel planning, ticketing, tourism services"},

    # Personal Services
    {"title": "Tailor / Fashion Designer", "description": "tailoring, garment making, fashion design, stitching"},
    {"title": "Beautician / Salon Staff", "description": "beauty services, salon work, makeup, hairstyling"},
    {"title": "Fitness Trainer", "description": "personal training, gym coaching, fitness instruction"},
    {"title": "Security Guard", "description": "security services, surveillance, premises protection"},
    {"title": "Housekeeping Staff", "description": "cleaning, housekeeping, facility maintenance"},

    # Academia / Teaching / Law
    {"title": "Teacher / Instructor", "description": "teaching, curriculum design, classroom instruction, tutoring"},
    {"title": "University Lecturer", "description": "higher education teaching, academic research, university courses"},
    {"title": "Research Assistant", "description": "academic research, data collection, literature review"},
    {"title": "Librarian", "description": "library management, cataloging, information science"},
    {"title": "Lawyer / Legal Associate", "description": "legal practice, litigation, contracts, legal advice"},
    {"title": "Paralegal", "description": "legal research, documentation, supporting lawyers"},
    {"title": "Translator / Interpreter", "description": "language translation, interpretation services"},

    # Agriculture
    {"title": "Agricultural Officer", "description": "farming operations, crop management, agricultural extension"},
    {"title": "Veterinary Assistant", "description": "animal care support, livestock management"},
]

_model = None
_role_embeddings = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_role_embeddings():
    """Computes role embeddings once and caches them in memory for reuse across requests."""
    global _role_embeddings
    if _role_embeddings is None:
        model = get_model()
        role_texts = [f"{r['title']}: {r['description']}" for r in ROLES]
        _role_embeddings = model.encode(role_texts, convert_to_tensor=True)
    return _role_embeddings


def list_role_titles() -> list[str]:
    """Returns just the titles, for populating a frontend dropdown."""
    return [r["title"] for r in ROLES]


def resolve_role(user_input: str) -> dict:
    """
    Takes free-text or dropdown role input and returns the closest matching
    known role, along with a similarity score and a confidence flag.
    """
    user_input_clean = user_input.strip().lower()

    # Fast path: exact title match, no embedding computation needed
    for role in ROLES:
        if user_input_clean == role["title"].lower():
            return {
                "matched_role": role["title"],
                "similarity": 1.0,
                "source": "exact_match",
                "confident": True,
            }

    model = get_model()
    role_embeddings = get_role_embeddings()
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    similarities = util.cos_sim(user_embedding, role_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    CONFIDENCE_THRESHOLD = 0.35  # below this, the match is unreliable

    return {
        "matched_role": ROLES[best_idx]["title"],
        "similarity": round(best_score, 3),
        "source": "embedding_match",
        "confident": best_score >= CONFIDENCE_THRESHOLD,
    }