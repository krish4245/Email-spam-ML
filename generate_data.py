import pandas as pd
import random
import os
import re

# Set random seed for reproducibility
random.seed(42)

# Helper lists for placeholders
NAMES = ["John Smith", "Alice Johnson", "David Miller", "Sarah Connor", "Michael Brown", "Emma Watson", "Robert Davies",
         "Jessica Taylor", "William Wilson", "Olivia Thomas", "James Anderson", "Sophia Martinez", "Benjamin White",
         "Isabella Lopez", "Lucas Harris", "Mia Clark", "Henry Lewis", "Charlotte Robinson", "Alexander Walker",
         "Amelia Young", "Daniel Allen", "Harper King", "Matthew Wright", "Evelyn Scott", "Joseph Green"]

COMPANIES = ["Amazon", "Google", "Microsoft", "Apple", "Netflix", "Meta", "Adobe", "Salesforce", "Oracle", "IBM",
             "Intel", "Cisco", "Nvidia", "PayPal", "Stripe", "Zoom", "Slack", "Uber", "Airbnb", "Spotify",
             "Walmart", "Target", "Costco", "FedEx", "UPS", "Bank of America", "Chase", "Wells Fargo", "Citibank"]

PRODUCTS = ["Cloud Suite", "Enterprise Server", "Developer Pro License", "Wireless Headphones", "Smart Fitness Watch",
            "Premium Subscription", "Database Sync Tool", "CyberSecurity Shield", "AI Analytics Dashboard",
            "Marketing Automation Pack", "Office Chair Pro", "Dual Monitor Setup", "Noise Cancelling Earbuds",
            "Coffee Machine Elite", "Virtual Private Server"]

DEPARTMENTS = ["Human Resources", "Engineering Team", "Information Technology", "Sales & Marketing", "Finance Department",
               "Customer Support", "Operations Division", "Product Management", "Legal & Compliance", "Security Team"]

DOMAINS = ["company-internal.com", "mybusiness.org", "enterprise-corp.net", "localfirm.io", "techstartup.co"]
BAD_DOMAINS = ["verify-account-security-update.ru", "paypal-restore-access-now.cn", "chase-login-portal-ref38.net",
               "bonus-winner-lottery-claims.info", "offshore-funds-transfer-vault.cc", "secure-bank-login-info.xyz"]

SUBJECTS_SAFE = [
    "Meeting reschedule: Project kickoff", "Quarterly budget review", "Feedback requested: New UI draft",
    "Welcome to the team!", "Out of office notice", "Weekly status report updates", "Discussion on product roadmap",
    "Notes from today's sync", "Question regarding server deployment", "Draft proposal for review",
    "Client presentation deck attached", "Performance review documentation", "Holiday schedule announcement",
    "Collaboration on tech stack definition", "Design system alignment meeting"
]

# Curated pools of templates for each class.
# Each template will contain placeholders like {name}, {company}, {product}, {dept}, {domain}, {amount}, {date}, {url} etc.

SAFE_TEMPLATES = [
    "Hi {name},\n\nJust wanted to follow up on our discussion yesterday. I've updated the {product} proposal. Can you review the draft when you get a chance? Let me know what you think.\n\nBest,\n{sender}\n{dept}",
    "Hi team,\n\nThis is a quick reminder that our weekly {dept} sync has been moved to {time} tomorrow in Room B. Please update your status reports before the meeting.\n\nThanks,\n{sender}",
    "Dear employees,\n\nPlease note that the main office will be closed on {date} for the national holiday. Have a wonderful weekend and enjoy the time off with your families.\n\nBest regards,\n{company} {dept}",
    "Hi {name},\n\nCan we schedule a 15-minute call today to discuss the integration issues with {product}? Let me know your availability.\n\nThanks,\n{sender}",
    "Hey {name},\n\nGreat job on the presentation today! The client was really impressed with our progress on the {product}. Let's keep this momentum going.\n\nCheers,\n{sender}",
    "Hello,\n\nPlease find attached the minutes from today's {dept} alignment meeting. Let me know if I missed any action items or points.\n\nRegards,\n{sender}",
    "Hi {name},\n\nI have reviewed the code changes you submitted for {product}. Everything looks solid, I have approved the pull request. Let's deploy to staging after the tests pass.\n\nThanks,\n{sender}",
    "Hi all,\n\nThe system maintenance is scheduled for tonight starting at 10 PM. We expect about 30 minutes of downtime for {product}. Please save your work accordingly.\n\nThanks for your cooperation,\n{dept}",
    "Hi {name},\n\nAre you free for lunch today? I wanted to catch up on the hiring pipeline for the junior engineer role in {dept}.\n\nBest,\n{sender}",
    "Hello {name},\n\nHere is the documentation for the API endpoints of {product}. Let me know if you run into any issues while integrating it with your module.\n\nRegards,\n{sender}",
    "Hi team,\n\nI'm excited to announce that {name} has joined {company} as our new Lead Architect. Please join me in welcoming them during the team standup today!\n\nBest,\n{sender}",
    "Dear {name},\n\nYour feedback on the new software design has been incorporated into the project plan. The updated file is available on the shared drive. Let me know if you have further suggestions.\n\nRegards,\n{sender}",
    "Hi {name},\n\nThanks for sending over the project timeline. I have shared it with the client and they are comfortable with the milestones. We will start execution next week.\n\nBest,\n{sender}",
    "Hello,\n\nI will be out of the office starting {date} with limited access to email. For urgent matters, please contact {name} from the {dept} team.\n\nThanks,\n{sender}",
    "Hi team,\n\nCongratulations on hit our sales target for {product}! Special thanks to the engineering and QA teams for delivering a stable release.\n\nBest regards,\n{sender}\nVP of {dept}"
]

SPAM_TEMPLATES = [
    "CONGRATULATIONS!!! You have been selected as the lucky winner of a brand new iPad Pro! Click {url} to claim your prize immediately! Limited stock available, act now!",
    "Lose up to 15 pounds in just 2 weeks with our revolutionary new keto supplement! 100% natural, no side effects. Get your risk-free trial bottle today at {url}!",
    "Dear Friend, do you want to earn $5,000 a week from the comfort of your own home? No experience required! Just click here {url} to register for this amazing opportunity!",
    "Get cheap meds online! No prescription needed! Viagra, Cialis, Levitra and more at up to 80% discount! Fast shipping, discrete packaging. Order now at {url}",
    "RE: Your outstanding credit card debt can be wiped out completely! Lower your monthly payments by up to 60%. Speak to our specialists today at {url}. Don't wait!",
    "Amazing deals! Save up to 90% on top luxury brands like Rolex, Gucci, and Louis Vuitton. Click {url} to shop our exclusive clearance sale now!",
    "Hi, I saw your profile online and wanted to connect. Check out my private photos and webcam show here: {url}. It's completely free to join!",
    "Stop paying too much for car insurance! Compare rates from 20 top providers in less than 2 minutes and save up to $500 a year. Get your free quote at {url}",
    "URGENT: Your computer is infected with 14 critical viruses! Clean your registry and boost PC performance by 300% now. Download our antivirus tool for free at {url}",
    "Grow your business instantly with 10 million targeted email leads! Only $29.99 for a limited time. Boost sales and traffic. Buy the list now at {url}",
    "Discover the secret to making millions in the cryptocurrency market. Our automated trading bot guarantees a 95% win rate. Sign up today and watch your money grow! {url}",
    "Double your vocabulary in 30 days! Guaranteed results. Ideal for students, professionals, and job seekers. Click here to download the free audio program: {url}",
    "Get a university degree in less than 15 days based on your work experience! Bachelor, Master, PhD available. No exams, no classes. Visit {url} to apply now!",
    "Special Promotion: Refinance your home loan today at historical low interest rates. Pay off your mortgage faster. Check your eligibility instantly at {url}",
    "You have won a free $500 Amazon gift card! Only 3 surveys away from claiming your reward. Click here {url} to start the short questionnaire now!"
]

PHISHING_TEMPLATES = [
    "Dear {company} Customer,\n\nWe detected suspicious login attempts to your account from an unknown IP address. For your security, we have temporarily suspended your account. To restore access, please verify your credentials immediately at {url} within 24 hours.",
    "URGENT: Security Update Required!\n\nDear User, our technical team is upgrading our database servers. You must update your login credentials to avoid service termination. Please log in to your account here: {url} to confirm your profile.",
    "Notification of Tax Refund!\n\nAfter our annual calculations of your fiscal activity, we have determined that you are eligible to receive a tax refund of {amount}. Please submit your bank details online to process the refund: {url}",
    "Your PayPal Account Has Been Locked!\n\nDear Client, we noticed unauthorized charges on your account. To secure your funds, we have locked your profile. Please click here {url} to verify your identity and unlock your account.",
    "Microsoft Office 365: Action Required!\n\nYour password for {email} will expire today. To retain your emails and files, please keep your current password by verifying your credentials at the security portal: {url}",
    "DHL Express Delivery Notification\n\nYour package could not be delivered due to an incorrect address. A shipping fee of $1.50 is outstanding. Please update your delivery address and pay the fee at {url} to schedule redelivery.",
    "HR Impersonation: Important Document\n\nDear Colleague,\n\nPlease review the updated Employee Handbook and remote work guidelines for {company}. You are required to sign the acknowledgment form by logging in here: {url}",
    "Immediate Wire Transfer Request\n\nKrishna, I am in a meeting and need you to process an urgent wire transfer of {amount} to our new vendor today. The invoice details are here: {url}. Please let me know once done. - Sent from my iPhone",
    "Security Alert: Google Account Compromised!\n\nSomeone recently used your password to try to sign in to your Google Account from a device in Russia. If this wasn't you, please secure your account by changing your password immediately: {url}",
    "Netflix: Update Your Payment Method\n\nWe were unable to process your monthly subscription fee. Your membership will be canceled if you do not update your billing info. Update payment now at: {url}",
    "Bank of America Security Alert\n\nDear Valued Customer, a transfer of {amount} has been initiated from your account. If you did not authorize this transaction, please cancel it immediately at our security portal: {url}",
    "Wells Fargo Security Update\n\nWe require all debit card holders to update their PIN and contact information to comply with new federal security regulations. Please verify your card details here: {url}",
    "Your Apple ID has been suspended!\n\nSomeone attempted to purchase a subscription to iCloud using your account. Your Apple ID has been locked for security. To unlock, verify your identity: {url}",
    "HR: New Salary Structure Review\n\nAll employees of {company} are requested to review the new salary structure and bonus eligibility criteria for Q3. Access the secure portal to view your personalized compensation details: {url}",
    "IT Support: Urgent Security Patch\n\nA critical vulnerability has been discovered in {product}. You must log in to the enterprise network access control portal to apply the patch immediately: {url}"
]

PROMOTION_TEMPLATES = [
    "Hi {name},\n\nGet ready for the summer! ☀️ Enjoy up to 50% off on all items across our store this weekend only. Use code SUMMER50 at checkout. Free shipping on orders over $35! Shop now at {url}",
    "Hey there! 🎉 It's our anniversary, and we're celebrating with you! Take an extra 20% off our best-selling {product}. Join our VIP loyalty program to unlock double reward points. Click {url}",
    "Hi {name},\n\nIs your team looking for a better way to manage workflows? Try {product} free for 30 days. No credit card required. Streamline task tracking and collaborate in real-time. Start trial: {url}",
    "Introducing the all-new {product}! Engineered for speed and efficiency, it helps teams like yours save up to 10 hours a week. Order now and get early bird pricing — save $200 today! {url}",
    "Only 24 hours left! ⏳ Our Flash Sale is ending soon. Don't miss out on these incredible discounts on electronics, home decor, and fashion. Visit {url} now before everything sells out!",
    "Hey {name},\n\nWe noticed you left some items in your cart. To help you decide, here is a special coupon for 10% off: CART10. Click here to complete your order with free shipping: {url}",
    "Hi there,\n\nSubscribe to our weekly newsletter and stay updated with the latest trends in tech, design, and product management. Plus, get a free copy of our e-book 'Productivity Hacks'. Sign up: {url}",
    "Hey {name},\n\nRefer a friend to {company} and you both get a $20 credit when they make their first purchase! Sharing is caring. Grab your unique referral link here: {url}",
    "Upgrade to Premium today and unlock unlimited storage, advanced analytics, and priority customer support for {product}. Apply code PREMIUM30 for 30% off your first year. Upgrade: {url}",
    "Hi {name},\n\nWe are excited to invite you to our upcoming webinar: 'Scaling engineering teams with {product}'. Learn from industry experts and get a live demo. Register for free: {url}",
    "Big news! 📣 We've just launched major updates for {product}, including dark mode, custom integrations, and faster load times. Read our blog post to see what's new: {url}",
    "Hey fashion lover, refresh your wardrobe with our new Spring Collection. Vibrant colors, breathable fabrics, and modern fits. Browse the catalog and get 15% off your first order: {url}",
    "Hi {name},\n\nAs a valued member of our loyalty program, we are giving you early access to our Black Friday deals. Shop the discounts 48 hours before anyone else! Access sale: {url}",
    "Hey there! Looking to learn a new skill? Get 70% off on all online courses this month. Coding, photography, marketing, business and more. Browse classes: {url}",
    "Dear client,\n\nLooking for the perfect gift? Buy a gift card of $100 and get an extra $20 voucher for yourself. The promotion runs until the end of the month. Order online: {url}"
]

SUSPICIOUS_TEMPLATES = [
    "Dearest One,\n\nI am Mrs. Sarah Williams from Kuwait. I am suffering from long-term breast cancer and have decided to donate my late husband's wealth of $5.5 million to an honest person who will use it for charity. Please reply to my private email: {email} for details.",
    "CONFIDENTIAL BUSINESS PROPOSAL\n\nDear Friend, I am the director of auditing at Bank of Africa. During our last audit, we discovered an abandoned account belonging to a deceased foreign investor with a balance of {amount}. I need your assistance to transfer this fund out of the country. We will share 40/60. Please reply urgently.",
    "URGENT ASSISTANCE REQUIRED\n\nHello, I am a barrister representing a client who died in a car crash without leaving a will. You share the same last name, and I want to present you as the next of kin to inherit their estate valued at $12.5 million. Please contact me at {email} for a secret discussion.",
    "Attention: Beneficiary,\n\nThis is to inform you that the United Nations Compensation Commission has approved the payment of {amount} to you as compensation for fraud victims. Your fund has been loaded onto an ATM Visa card. Please provide your delivery address and phone number to courier it to you.",
    "Dear Partner,\n\nI have a high-yield investment opportunity in crude oil trading that guarantees a 400% return in 30 days. This is highly confidential and risk-free. Minimum investment is $5,000 in Bitcoin. Please download the prospectus: {url} and reply with your decision.",
    "PRIVATE AND CONFIDENTIAL\n\nI am writing to you regarding a mutual business interest. I have access to sensitive information about an upcoming corporate acquisition. We can make significant profits if we act quickly. Please contact me via encrypted email: {email}. Do not mention this to anyone.",
    "Urgent response needed,\n\nAre you available? I need you to do a quick task for me. I am currently tied up in an important board meeting and can't take calls. I need you to purchase 5 Google Play gift cards of $100 each and email me the codes immediately. I will reimburse you tonight. Reply ASAP.",
    "Dear sir/madam,\n\nI represent a charity foundation in West Africa. We are seeking international partners to distribute relief materials and funds. If you are interested in coordinating this project in your region, please reply with your resume and contact info to {email}.",
    "CRITICAL WARNING!\n\nWe have hacked your computer system and extracted all your private browsing history, webcam recordings, and personal files. If you do not pay $1,000 in Bitcoin to our wallet address within 48 hours, we will release this content to all your email contacts. Do not reply to this email, just pay.",
    "RE: Business partnership proposal,\n\nWe have reviewed your company profile and believe there is a strong synergy between our firms. Please open the attached zip archive containing our project requirements and pricing proposal: {url}. We look forward to your prompt response.",
    "Dear Friend,\n\nI am a diplomat at the United Nations. I have been assigned to deliver a consignment box containing $8.5 million to your country. I am currently held at the airport customs due to clearance fees of $2,500. Please assist with the fee and I will deliver the box to you. Reply to {email}.",
    "Request for quotation - Urgent,\n\nPlease provide your best price and delivery terms for the items listed in the attached purchase order. The order needs to be shipped next week. Access the purchase order here: {url} and submit your bid.",
    "Hello Krishna,\n\nI need you to review this confidential financial report before my meeting with the investors at 3 PM. It contains sensitive projections. Please do not share it outside the executive group. Report link: {url}",
    "URGENT: Outstanding Invoice #83920\n\nDear Customer, your payment for invoice #83920 is overdue by 15 days. A late fee of 5% has been applied. Please download the attached invoice and pay immediately to avoid service interruption. Link: {url}",
    "Romance & Trust: A letter to my love,\n\nMy dear, I was so happy to receive your message. I am currently deployed overseas and missing home. I can't wait to be with you. I want to send you some luxury gifts, but I need you to pay the courier customs charge of $450. Please send the fee via Western Union today."
]

def generate_samples(templates, count, label):
    """Generate randomized email text samples from templates."""
    samples = []
    
    # Pre-compiled list of possible components for generation
    sender_firsts = ["Sarah", "John", "David", "Michael", "Emma", "Jessica", "Daniel", "James", "Robert", "Linda", "Karen", "Thomas"]
    sender_lasts = ["Miller", "Smith", "Jones", "Davis", "Wilson", "Brown", "Taylor", "Anderson", "Thomas", "White"]
    
    url_schemes = ["http://", "https://"]
    legit_paths = ["/login", "/dashboard", "/verify", "/update", "/billing", "/newsletter", "/webinar", "/catalog"]
    scam_paths = ["/verify-identity/paypal", "/secure-login-chase/index.html", "/claim-prize-lottery-88", "/bitcoin-wallet-deposit"]
    
    for i in range(count):
        # Pick template
        template = random.choice(templates)
        
        # Pick template elements
        name = random.choice(NAMES)
        sender = f"{random.choice(sender_firsts)} {random.choice(sender_lasts)}"
        company = random.choice(COMPANIES)
        product = random.choice(PRODUCTS)
        dept = random.choice(DEPARTMENTS)
        
        # Domains
        domain = random.choice(DOMAINS)
        bad_domain = random.choice(BAD_DOMAINS)
        
        # Sender email address
        sender_user = sender.lower().replace(" ", ".")
        if label in ["phishing", "spam", "suspicious"]:
            email = f"{sender_user}@{bad_domain}"
        else:
            email = f"{sender_user}@{domain}"
            
        # Amount
        amount = f"${random.randint(100, 50000):,}"
        
        # Dates and times
        date = f"{random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])} {random.randint(1, 28)} Oct"
        time = f"{random.randint(9, 17)}:00 {random.choice(['AM', 'PM'])}"
        
        # URLs
        if label in ["safe", "promotion"]:
            # Legitimate URL
            url = f"{random.choice(url_schemes)}{company.lower()}.com{random.choice(legit_paths)}"
        else:
            # Dangerous URL or URL shortener
            if random.random() < 0.4:
                # URL Shortener
                shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd"]
                url = f"{random.choice(url_schemes)}{random.choice(shorteners)}/{random.randint(10000, 99999)}"
            else:
                # Scam/Suspicious TLD URL
                url = f"{random.choice(url_schemes)}{bad_domain}{random.choice(scam_paths)}"
                
        # Fill template
        filled_text = template.format(
            name=name,
            sender=sender,
            company=company,
            product=product,
            dept=dept,
            email=email,
            amount=amount,
            date=date,
            time=time,
            url=url
        )
        
        # Sometimes prepend a subject line for realism
        if random.random() < 0.6:
            if label == "safe":
                subject = random.choice(SUBJECTS_SAFE)
            elif label == "spam":
                subject = f"WINNER: {random.choice(['You won!', 'Claim your reward', 'Incredible prize opportunity', 'Congratulations!'])}"
            elif label == "phishing":
                subject = f"URGENT: {random.choice(['Security alert', 'Account suspended', 'Verify payment details', 'Action required immediately'])}"
            elif label == "promotion":
                subject = f"SPECIAL OFFER: {random.choice(['X% off all orders', 'Get 50% discount today', 'Exclusive deal inside', 'Introduce our new product'])}"
            else: # suspicious
                subject = f"CONFIDENTIAL: {random.choice(['Urgent assistance requested', 'Business partnership proposal', 'Strictly private', 'Wire transfer details'])}"
                
            filled_text = f"Subject: {subject}\n\n{filled_text}"
            
        # Add random noise (typos, casing) in some messages (mostly spam/phishing/suspicious)
        if label in ["spam", "phishing", "suspicious"] and random.random() < 0.3:
            # Let's double some exclamation points
            filled_text = filled_text.replace("!", "!!!")
            # Sometimes capitalize entire words randomly
            words = filled_text.split()
            for idx in range(min(5, len(words))):
                rand_idx = random.randint(0, len(words)-1)
                if len(words[rand_idx]) > 3:
                    words[rand_idx] = words[rand_idx].upper()
            filled_text = " ".join(words)
            
        samples.append(filled_text)
        
    return samples

def main():
    print("Initializing Synthetic Email Data Generator...")
    
    # Target counts:
    # safe: 5000
    # spam: 5000
    # phishing: 5000
    # promotion: 5000
    # suspicious: 3000
    targets = {
        "safe": 5000,
        "spam": 5000,
        "phishing": 5000,
        "promotion": 5000,
        "suspicious": 3500  # Target: 3000+
    }
    
    data = []
    
    # Generate Safe
    print("Generating 'Safe' emails...")
    safe_emails = generate_samples(SAFE_TEMPLATES, targets["safe"], "safe")
    for email in safe_emails:
        data.append({"text": email, "label": "safe"})
        
    # Generate Spam
    print("Generating 'Spam' emails...")
    spam_emails = generate_samples(SPAM_TEMPLATES, targets["spam"], "spam")
    for email in spam_emails:
        data.append({"text": email, "label": "spam"})
        
    # Generate Phishing
    print("Generating 'Phishing' emails...")
    phishing_emails = generate_samples(PHISHING_TEMPLATES, targets["phishing"], "phishing")
    for email in phishing_emails:
        data.append({"text": email, "label": "phishing"})
        
    # Generate Promotion
    print("Generating 'Promotion' emails...")
    promo_emails = generate_samples(PROMOTION_TEMPLATES, targets["promotion"], "promotion")
    for email in promo_emails:
        data.append({"text": email, "label": "promotion"})
        
    # Generate Suspicious
    print("Generating 'Suspicious' emails...")
    susp_emails = generate_samples(SUSPICIOUS_TEMPLATES, targets["suspicious"], "suspicious")
    for email in susp_emails:
        data.append({"text": email, "label": "suspicious"})
        
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    # Save CSV
    output_path = "data/processed/emails_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print("\nDataset Generation Complete!")
    print(f"Total Emails Generated: {len(df)}")
    print(df["label"].value_counts())
    print(f"Dataset successfully saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
