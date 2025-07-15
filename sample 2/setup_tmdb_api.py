# setup_tmdb_api.py
import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "./Dataset",
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")

def check_env_file():
    """Check if .env file exists"""
    if os.path.exists(".env"):
        print("✅ .env file exists")
        return True
    else:
        print("❌ .env file not found")
        return False

def build_dataset():
    """Build the TMDB dataset"""
    print("🎬 Building TMDB dataset...")
    print("   This may take 10-15 minutes...")
    
    try:
        result = subprocess.run([sys.executable, "build_tmdb_dataset.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dataset built successfully!")
            return True
        else:
            print(f"❌ Error building dataset: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error building dataset: {e}")
        return False

def check_dataset():
    """Check if dataset exists"""
    dataset_path = "./Dataset/tmdb_movies.csv"
    if os.path.exists(dataset_path):
        print(f"✅ Dataset exists at {dataset_path}")
        return True
    else:
        print(f"❌ Dataset not found at {dataset_path}")
        return False

def run_tests():
    """Run API tests"""
    print("🧪 Running API tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_api.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Tests passed!")
            return True
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up TMDB Recommendation API...")
    print("=" * 50)
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Create directories
    create_directory_structure()
    success_count += 1
    
    # Step 2: Check .env file
    if check_env_file():
        success_count += 1
    else:
        print("   Please create a .env file with your TMDB API key")
    
    # Step 3: Install requirements
    if install_requirements():
        success_count += 1
    
    # Step 4: Build dataset or check if exists
    if check_dataset():
        print("   Dataset already exists, skipping build...")
        success_count += 1
    else:
        if build_dataset():
            success_count += 1
    
    # Step 5: Check if we can start the API
    if success_count >= 4:
        print("🎉 Setup completed successfully!")
        success_count += 1
    else:
        print("⚠️  Setup completed with some issues")
    
    print("\n" + "=" * 50)
    print(f"Setup Status: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("\n📋 Next steps:")
        print("   1. Run the API: python tmdb_recommendation_api.py")
        print("   2. Open your browser: http://localhost:8000")
        print("   3. API docs: http://localhost:8000/docs")
        print("   4. Test endpoint: http://localhost:8000/recommend")
        print("\n💡 Example API call:")
        print("   POST http://localhost:8000/recommend")
        print("   Body: {\"title\": \"Superman\", \"n_recommendations\": 10}")
        print("\n🧪 To run tests: python test_api.py")
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()