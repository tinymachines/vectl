#!/usr/bin/env python3
"""
Test the enhanced JSON repair on the provided samples
"""

import json
import os
from enhanced_json_repair import EnhancedJSONRepair

# Create output directory
os.makedirs("repaired_samples", exist_ok=True)

# Initialize the repair tool
repair_tool = EnhancedJSONRepair(enable_llm_repair=True)

# Sample 1 - Missing sections
sample1 = """
{
    "title": "PAYROLL AUTHORIZATION FORM",
    "subtitles": [
        "(Please Use Typewriter or Ballpoint Pen)",
        "U.S. HOUSE OF REPRESENTATIVES",
        "Washington, D.C. 20515"
    ],
    "sections": [
        {
            "title": "To the Clerk of the House of Representatives:",
            "text": "I hereby authorize the following payroll action:"
        },
        {
            "title": "Employee Name (First-Middle-Last)",
            "text": "Ann Farnald Taylor"
        },
        {
            "title": "Employee Social Security Number",
            "text": "036-34-9167"
        },
        {
            "title": "Employing Office or Committee/Subcommittee",
            "text": "Assassination"
        },
        {
            "title": "Position Title",
            "text": "Researcher"
        },
        {
            "title": "Gross Annual Salary*",
            "text": "$19,300"
        },
        {
            "title": "Researcher",
        },
        {
            "title": "I certify that this authorization is not in violation of 5 U.S.C. 3110(b), prohibiting the employment of relatives.",
            "text": "Date: July 22, 1978"
        },
        {
            "title": "LOUIS STOKES",
            "text": "Chairman"
        },
        {
            "title": "CHALLENGES",
        },
        {
            "title": "LOUIS STOKES",
            "text": "Chairman"
        },
        {
            "title": "CHALLENGES",
        },
        {
            "title": "Office of Finance use only:",
            "text": "Monthly Annuity $_______00 as of"
        },
        {
            "title": "Copy for Initiating Office or Committee",
        }
    ]
}
"""

# Sample 2 - Truncated content
sample2 = """
{
  "header": {
    "classification": "SECRET"
  },
  "body": [
    {
      "text": "Cassasin said that type of special design plant dealt with aviation, nuclear energy, bacteriological warfare, etc. He said that he cannot remember the exact location of the plant but he remembered the city was of interest for that reason. Additionally, anyone in the area of those plants was of interest to the agency. Cassasin said he is confident that Oswald was not working in any type of security facility. He said even with the presence of the design plants, Minsk was not considered a sensitive area. He also said that he believes they had some type of encyclopedic information at the agency on the radio factory in Minsk where Oswald worked. He said that kind of information was maintained in the Voice of America Research and Reporting. Cassasin said he was not aware of a KGB facility in Minsk."
    },
    {
      "text": "Cassasin said the Legal Travellers Program was headed at the time of his memo by Alexander Bokaloff (phonetic). He said the program began pre-1960 with the lessening of restrictions between the United States and the Soviet Union. He said Bokaloff had some of the Soviet Union with his parents as a child. He explained that defectors such as Bokaloff were kept at arms length for security reasons by the agency."
    },
    {
      "text": "The Legal Travellers Program operated in such a way that the agency contact would tou..."
    }
  ]
}
"""

# Sample 3 - Missing section declarations
sample3 = """
{
    "title": "PAYROLL AUTHORIZATION FORM",
    "sub_title": "U.S. HOUSE OF REPRESENTATIVES",
    "sub_sub_title": "Washington, D.C. 20515",
    "section_1": {
        "title": "To the Clerk of the House of Representatives:",
        "text": "I hereby authorize the following payroll action:"
    },
        "title": "Employee Name (First-Middle-Last)",
        "text": "I. Charles Mathis"
    },
    "section_3": {
        "title": "Effective Date",
        "text": "March 1, 1978"
    },
        "title": "Type of Action",
        "text": "Appointment"
    },
    "section_5": {
        "title": "Employing Office or Committee/Subcommittee",
        "text": "Assessment and Taxation"
    },
        "title": "Position Title",
        "text": "Special Counsel"
    },
    "section_7": {
        "title": "Gross Annual Salary",
        "text": "$30,000"
    },
        "title": "Special Investigative Staff of Standing Committee or Select Committee Authority—H. Res. 956 of 93rd Congress",
        "text": "Joint Committee"
    },
    "section_9": {
        "title": "Position Number",
        "text": "If applicable, Level"
    },
        "title": "Step",
    },
    "section_11": {
        "text": "March 14, 1978"
    },
        "title": "Chairman",
        "text": "Louis Stokes"
    },
    "section_13": {
        "title": "Office of Finance use only:",
        "text": "Office Code"
    },
        "title": "Monthly Annuity $",
        "text": "00 as of"
    },
    "section_15": {
        "title": "Copy for Initiating Office or Committee",
    }
}
"""

# Sample 4 - Comments in JSON
sample4 = """
{
  "header": {
    "date": "03/04/77",
    "report_id": "001-00-00",
    "office": "R3000. SELECT COMMITTEE ON ASSASSINATIONS"
  },
  "body": [
    {
      "employee_name": "ACOSTA, LINDA RAY",
      "social_security_number": "416-54-2968",
      "annual_salary": "12,000.00",
      "gross_pay": "1,025.00",
      "remarks": "TERMINATED 03-08-77"
    },
    {
      "employee_name": "ADAMS, EDITH",
      "social_security_number": "169-16-0121",
      "annual_salary": "11,070.00",
      "gross_pay": "246.00",
      "remarks": "TERMINATED 03-08-77"
    },
    // ... (rest of the employees' information)
  ],
  "footer": {
    "notes": "Extracted under the 50th Anniversary Freedom of Information Act Request. Request Number: 2017-001-FOIA. Requester: [Redacted]. Date: 2017-01-01."
  }
}
"""

# Sample 5 - Invalid key structure
sample5 = """
{
  "JFK ASSASSINATION SYSTEM": {
    "IDENTIFICATION FORM": {
      "AGENCY INFORMATION": {
        "AGENCY": "HSCA",
        "RECORD NUMBER": "150-10070-10163",
        "RECORDS SERIES": "STAFF PAYROLL RECORDS",
        "AGENCY FILE NUMBER": "95"
      },
      "DOCUMENT INFORMATION": {
        "ORIGINATOR": "HSCA",
        "FROM": "TO",
        "TITLE": "",
        "DATE": "01/01/77",
        "PAGES": "14",
        "SUBJECTS": "HSCA; ADMINISTRATION",
        "TAYLOR, ANN FUNNA"
      },
      "DOCUMENT TYPE": "PRINTED FORM",
      "CLASSIFICATION": "P",
      "RESTRICTIONS": "3",
      "CURRENT STATUS": "P",
      "DATE OF LAST REVIEW": "07/16/93",
      "OPENING CRITERIA": "",
      "COMMENTS": "",
      "Box 3.": ""
    }
  }
}
"""

# Sample 6 - Empty fields
sample6 = """
{
  "title": "PAYROLL AUTHORIZATION FORM",
  "subtitles": [
    "U.S. HOUSE OF REPRESENTATIVES",
    "Washington, D.C. 20515"
  ],
  "sections": [
    {
      "title": "To the Clerk of the House of Representatives:",
      "body": "I hereby authorize the following payroll action."
    },
    {
      "title": "Employee Name (First-Middle-Last)",
      "body": "Louis H. Hinds'le"
    },
    {
      "title": "Employee Social Security Number",
      "body": "219 58 7593"
    },
    {
      "title": "Employing Office or Committee",
    },
    {
      "title": "Assassinations",
      "body": "Assassinations"
    }
  ]
}
"""

# Sample 7 - Truncated JSON with object termination issue
sample7 = """
{
  "header": {
    "date": "November 15, 1978",
    "sender": {
      "name": "United States House of Representatives",
      "committee": "Select Committee on Assassinations",
      "office": "Personnel Office",
      "location": "Washington, D.C."
    },
    "recipient": {
      "name": "Maurice Israel",
      "ssn": "128-20-8996"
    }
  },
  "verification": {
    "by": {
      "name": "Mildred Houston",
      "title": "Director, Background Investigation Section"
    }
  },
  "footer": {
    "signatures": [
      {
        "name": "Stanley N. Lupkin",
        "title": "Commissioner",
        "date": "11-20-78"
      }
    ],
    "signature": "[Signature]",
  }
}
"""

# Collect all samples
samples = [
    {"name": "sample1", "content": sample1},
    {"name": "sample2", "content": sample2},
    {"name": "sample3", "content": sample3},
    {"name": "sample4", "content": sample4},
    {"name": "sample5", "content": sample5},
    {"name": "sample6", "content": sample6},
    {"name": "sample7", "content": sample7}
]

# Test each sample
for i, sample in enumerate(samples):
    name = sample["name"]
    content = sample["content"]
    
    print(f"\nProcessing {name}...")
    
    # Try to parse without repair
    try:
        json.loads(content)
        print(f"  {name} parses successfully without repair")
        valid = True
    except json.JSONDecodeError as e:
        print(f"  {name} fails to parse: {str(e)}")
        valid = False
    
    # Save the original sample
    with open(f"repaired_samples/{name}_original.json", "w", encoding="utf-8") as f:
        f.write(content)
    
    # If it's invalid, try to repair it
    if not valid:
        # Try basic repair
        basic_repaired = repair_tool._basic_repair(content)
        try:
            json.loads(basic_repaired)
            print(f"  {name} repaired successfully with basic repair")
            
            # Save the basic repaired version
            with open(f"repaired_samples/{name}_basic_repaired.json", "w", encoding="utf-8") as f:
                f.write(basic_repaired)
                
        except json.JSONDecodeError:
            print(f"  {name} could not be repaired with basic repair")
        
        # Try full repair
        repaired_obj, success, error = repair_tool.repair_json(content)
        
        if success:
            print(f"  {name} repaired successfully with full repair")
            
            # Save the fully repaired version
            with open(f"repaired_samples/{name}_full_repaired.json", "w", encoding="utf-8") as f:
                json.dump(repaired_obj, f, indent=2)
                
        else:
            print(f"  {name} could not be repaired: {error}")

print("\nAll samples processed and saved to repaired_samples/")
