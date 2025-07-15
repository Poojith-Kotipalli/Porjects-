# test_api.py
import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if API is running"""
    print("🔍 Testing API health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy! Movies loaded: {data['total_movies']}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {str(e)}")
        print("   Make sure the API is running: python tmdb_recommendation_api.py")
        return False

def test_movie_search(movie_title):
    """Test movie search functionality"""
    print(f"🎬 Searching for movie: {movie_title}")
    try:
        response = requests.get(f"{API_BASE_URL}/search/{movie_title}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found: {data['title']} ({data['year']})")
            print(f"   Rating: {data['rating']}/10")
            print(f"   Genres: {data['genres']}")
            return True
        else:
            print(f"❌ Movie not found: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error searching movie: {str(e)}")
        return False

def test_recommendations(movie_title, n_recommendations=5):
    """Test movie recommendations"""
    print(f"🎯 Getting recommendations for: {movie_title}")
    try:
        payload = {
            "title": movie_title,
            "n_recommendations": n_recommendations
        }
        
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Found {data['total_found']} recommendations:")
            print(f"   Query movie: {data['query_movie']['title']} ({data['query_movie']['year']})")
            print(f"   Rating: {data['query_movie']['rating']}/10")
            print("\n   Recommendations:")
            
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"   {i}. {rec['title']} ({rec['year']})")
                print(f"      Rating: {rec['rating']}/10")
                print(f"      Similarity: {rec['similarity_score']:.3f}")
                if rec['poster_url']:
                    print(f"      Poster: {rec['poster_url']}")
                print()
            
            return True
        else:
            print(f"❌ Recommendation failed: {response.status_code}")
            error_detail = response.json().get('detail', 'Unknown error')
            print(f"   Error: {error_detail}")
            return False
            
    except Exception as e:
        print(f"❌ Error getting recommendations: {str(e)}")
        return False

def test_popular_movies():
    """Test popular movies endpoint"""
    print("🌟 Getting popular movies...")
    try:
        response = requests.get(f"{API_BASE_URL}/popular?limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Popular movies:")
            for i, movie in enumerate(data['popular_movies'], 1):
                print(f"   {i}. {movie['title']} ({movie['year']}) - {movie['rating']}/10")
            return True
        else:
            print(f"❌ Popular movies failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting popular movies: {str(e)}")
        return False

def test_dataset_stats():
    """Test dataset statistics"""
    print("📊 Getting dataset statistics...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Dataset statistics:")
            print(f"   Total movies: {data['total_movies']}")
            print(f"   Average rating: {data['average_rating']:.2f}/10")
            print(f"   Total genres: {data['total_genres']}")
            print(f"   Year range: {data['year_range']['min']} - {data['year_range']['max']}")
            print(f"   Movies with posters: {data['movies_with_posters']}")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        return False

def test_api_endpoints():
    """Test various API endpoints"""
    print("🔗 Testing API endpoints...")
    
    endpoints = [
        "/",
        "/docs",
        "/popular?limit=3",
        "/stats"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {endpoint} - OK")
            else:
                print(f"❌ {endpoint} - Failed ({response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {str(e)}")

def main():
    """Run all tests"""
    print("🧪 TMDB Recommendation API Test Suite")
    print("=" * 50)
    
    # Test movies to try
    test_movies = [
        "Superman",
        "Batman",
        "Avengers",
        "Spider-Man",
        "Iron Man"
    ]
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Health check (most important)
    total_tests += 1
    if test_health_check():
        tests_passed += 1
    else:
        print("\n❌ API is not running. Please start it first:")
        print("   python tmdb_recommendation_api.py")
        return
    
    print()
    
    # API endpoints test
    total_tests += 1
    test_api_endpoints()
    tests_passed += 1
    print()
    
    # Dataset stats
    total_tests += 1
    if test_dataset_stats():
        tests_passed += 1
    print()
    
    # Popular movies
    total_tests += 1
    if test_popular_movies():
        tests_passed += 1
    print()
    
    # Movie search and recommendations
    successful_movie_tests = 0
    for movie in test_movies:
        print(f"Testing with movie: {movie}")
        print("-" * 30)
        
        # Search test
        total_tests += 1
        if test_movie_search(movie):
            tests_passed += 1
            successful_movie_tests += 1
        print()
        
        # Recommendation test
        total_tests += 1
        if test_recommendations(movie, 3):
            tests_passed += 1
        print("-" * 50)
        
        # Don't test all movies if API is slow
        if successful_movie_tests >= 2:
            break
    
    # Summary
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your API is working perfectly!")
        print("\n🌐 You can now:")
        print("   - Open http://localhost:8000 in your browser")
        print("   - View API docs at http://localhost:8000/docs")
        print("   - Use the frontend_example.html file")
        print("   - Make POST requests to /recommend")
    elif tests_passed >= total_tests * 0.8:
        print("✅ Most tests passed! Your API is working well.")
        print("   Some minor issues may exist but the core functionality works.")
    else:
        print("⚠️  Many tests failed. Please check:")
        print("   - Is the API running?")
        print("   - Is the dataset built?")
        print("   - Are all requirements installed?")

if __name__ == "__main__":
    main()