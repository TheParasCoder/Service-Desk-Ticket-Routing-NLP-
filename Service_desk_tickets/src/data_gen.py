import pandas as pd
import numpy as np
import os

def generate_data(num_samples=500):
    categories = ['Hardware', 'Billing', 'Access']
    
    # Templates for synthetic data generation
    templates = {
        'Hardware': [
            "My laptop is not turning on",
            "Keyboard keys are stuck",
            "Monitor is flickering",
            "Mouse not working",
            "Printer says paper jam but it's empty",
            "Need a replacement for my headset",
            "Computer making a loud noise",
            "Screen went black suddenly",
            "Docking station not connecting",
            "Webcam failed during meeting"
        ],
        'Billing': [
            "Invoice #12345 is incorrect",
            "Need to update credit card information",
            "Clarification on last month's charge",
            "Refund request for order #987",
            "Billing address needs to be changed",
            "Why was I charged twice?",
            "Payment failed notification",
            "Upgrade subscription plan cost",
            "Receipt for expense report",
            "Cancel my subscription renewal"
        ],
        'Access': [
            "Forgot my password",
            "Cannot log in to VPN",
            "Need access to the marketing folder",
            "Account locked out",
            "Reset 2FA settings",
            "Permission denied for Jira",
            "Error 403 trying to access dashboard",
            "New employee account setup",
            "Unable to connect to Wi-Fi",
            "SSO login is looping"
        ]
    }

    data = []
    
    for _ in range(num_samples):
        cat = np.random.choice(categories)
        base_text = np.random.choice(templates[cat])
        
        # Add some variation
        variations = [
            "",
            " Please help.",
            " Urgent!!",
            " It was working yesterday.",
            " I tried restarting but no luck.",
            " Thanks."
        ]
        variation = np.random.choice(variations)
        
        data.append({
            'text': base_text + variation,
            'label': cat
        })
        
    df = pd.DataFrame(data)
    return df

def main():
    print("Generating synthetic dataset...")
    df = generate_data(1000)
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Dataset generated in '{output_dir}/'. Train: {len(train_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    main()
