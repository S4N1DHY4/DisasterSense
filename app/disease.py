import pandas as pd

disaster_data = [
    ["Floods", "Cholera", "Waterborne", "Water purification, vaccination, hygiene measures"],
    ["Floods", "Typhoid Fever", "Fecal-oral", "Improved sanitation, vaccination, safe drinking water"],
    ["Floods", "Hepatitis A", "Fecal-oral", "Vaccination, improved sanitation, hygiene"],
    ["Floods", "Leptospirosis", "Waterborne", "Rodent control, water treatment, protective gear"],
    ["Floods", "Dengue Fever", "Vector-borne", "Vector control, mosquito repellents"],
    ["Floods", "Diarrheal Diseases", "Fecal-oral", "Improved sanitation, safe drinking water, hygiene practices"],
    ["Floods", "Gastroenteritis", "Fecal-oral", "Handwashing, water purification, safe food handling"],
    ["Floods", "Respiratory Infections", "Airborne", "Proper ventilation, respiratory hygiene, vaccination"],
    ["Tsunami", "Cholera", "Waterborne", "Water purification, vaccination, sanitation"],
    ["Tsunami", "Malaria", "Vector-borne", "Insecticide-treated nets, antimalarial drugs"],
    ["Tsunami", "Typhoid Fever", "Fecal-oral", "Improved sanitation, vaccination, safe drinking water"],
    ["Tsunami", "Hepatitis A", "Fecal-oral", "Vaccination, improved sanitation, hygiene"],
    ["Tsunami", "Dengue Fever", "Vector-borne", "Vector control, mosquito repellents"],
    ["Tsunami", "Leptospirosis", "Waterborne", "Rodent control, water treatment, protective gear"],
    ["Hurricane", "Cholera", "Waterborne", "Water purification, vaccination, sanitation"],
    ["Hurricane", "Malaria", "Vector-borne", "Insecticide-treated nets, antimalarial drugs"],
    ["Hurricane", "Typhoid Fever", "Fecal-oral", "Improved sanitation, vaccination, safe drinking water"],
    ["Hurricane", "Hepatitis A", "Fecal-oral", "Vaccination, improved sanitation, hygiene"],
    ["Hurricane", "Dengue Fever", "Vector-borne", "Vector control, mosquito repellents"],
    ["Hurricane", "Leptospirosis", "Waterborne", "Rodent control, water treatment, protective gear"],
    ["Earthquake", "Respiratory Infections", "Airborne", "Vaccination, hygiene, proper ventilation"],
    ["Earthquake", "Post-traumatic Stress Disorder (PTSD)", "Psychological trauma", "Psychological support, counseling"],
    ["Earthquake", "Diarrheal Diseases", "Fecal-oral", "Improved sanitation, safe drinking water, hygiene practices"],
    ["Earthquake", "Skin Infections", "Direct contact", "Good personal hygiene, wound care, antibiotics"],
    ["Tornado", "Respiratory Infections", "Airborne", "Vaccination, hygiene, proper ventilation"],
    ["Tornado", "Skin Infections", "Direct contact", "Good personal hygiene, wound care, antibiotics"],
    ["Tornado", "Traumatic Injuries", "Physical injury", "Safety measures, protective gear, first aid"],
    ["Fire", "Respiratory Infections", "Airborne", "Respiratory hygiene, smoke avoidance, vaccination"],
    ["Fire", "Skin Infections", "Direct contact", "Good personal hygiene, wound care, antibiotics"],
    ["Fire", "Heat Stroke", "Thermal stress", "Hydration, cooling, shade, clothing"],
    ["Fire", "Diarrheal Diseases", "Fecal-oral", "Improved sanitation, hygiene, safe drinking water"],
    ["Fire", "Post-traumatic Stress Disorder (PTSD)", "Psychological trauma", "Psychological support, counseling"],
    ["Volcanic Eruptions", "Respiratory Infections", "Airborne", "Respiratory hygiene, ash masks, vaccination"],
    ["Volcanic Eruptions", "Skin Infections", "Direct contact", "Good personal hygiene, wound care, antibiotics"],
    ["Volcanic Eruptions", "Traumatic Injuries", "Physical injury", "Safety measures, protective gear, first aid"],
    ["Blizzard", "Hypothermia", "Cold-related", "Proper clothing, shelter, heating measures"],
    ["Blizzard", "Frostbite", "Cold-related", "Proper clothing, shelter, heating measures"],
    ["Blizzard", "Respiratory Infections", "Airborne", "Vaccination, hygiene, proper ventilation"],
    ["Blizzard", "Traumatic Injuries", "Physical injury", "Safety measures, proper shelter"],
    ["Hailstorm", "Traumatic Injuries", "Physical injury", "Safety measures, protective gear"],
    ["Hailstorm", "Hypothermia", "Cold-related", "Proper clothing, shelter, heating measures"],
    ["Landslide", "Skin Infections", "Direct contact", "Good personal hygiene, wound care, antibiotics"],
    ["Landslide", "Respiratory Infections", "Airborne", "Vaccination, hygiene, proper ventilation"],
    ["Landslide", "Malaria", "Vector-borne", "Insecticide-treated nets, antimalarial drugs"],
    ["Floods", "Giardia", "Waterborne", "Water filtration, avoiding drinking contaminated water"],
    ["Floods", "Cryptosporidiosis", "Waterborne", "Water filtration, avoiding contaminated water"],
    ["Floods", "Schistosomiasis", "Waterborne", "Proper sanitation, avoiding contaminated water"]
]


df = pd.DataFrame(disaster_data, columns=["Disaster", "Disease", "Transmission Mode", "Prevention Measures"])

df.to_csv("data/dataset.csv", index=False)

print("Dataset saved as 'dataset.csv'.")
