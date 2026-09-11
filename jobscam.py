# JOB SCAM DETECTION - PYTHON DATA STRUCTURES

# LIST - Job titles
jobs = [
    "Software Developer",
    "Data Analyst",
    "Web Developer",
    "Data Entry Operator",
    "Python Developer"
]

print("JOB LIST:")
print(jobs)

# List methods
jobs.append("Java Developer")
jobs.insert(1, "ML Engineer")
jobs.remove("Web Developer")

print("After adding and removing:")
print(jobs)

print("Count:", jobs.count("Python Developer"))
print("Index:", jobs.index("Data Analyst"))

jobs.sort()
print("Sorted:", jobs)

jobs.reverse()
print("Reversed:", jobs)

removed_job = jobs.pop()
print("Popped:", removed_job)

new_jobs = jobs.copy()
print("Copied list:", new_jobs)


# DICTIONARY - Job details
job = {
    "title": "Software Developer",
    "company": "ABC Technologies",
    "location": "Calicut",
    "salary": 30000,
    "skills": "Python, Django, SQL",
    "fraudulent": 0
}

print("\nJOB DETAILS:")
print(job)

# Dictionary methods
print("Job title:", job.get("title"))

print("Keys:", job.keys())
print("Values:", job.values())
print("Items:", job.items())

# Update value
job["salary"] = 35000

# Add new value
job.update({"experience": "2 years"})

print("Updated job:")
print(job)

# Remove value
job.pop("experience")

print("After pop:")
print(job)


# SET - Unique skills
skills = {
    "Python",
    "Django",
    "SQL",
    "Python"
}

print("\nSKILLS:")
print(skills)

# Set methods
skills.add("Machine Learning")
print("After add:", skills)

skills.remove("SQL")
print("After remove:", skills)

skills.discard("Java")
print("After discard:", skills)

print("Python available:", "Python" in skills)


# FRAUD DETECTION
if job["fraudulent"] == 1:
    print("\nResult: This job is a SCAM")
else:
    print("\nResult: This job is GENUINE")