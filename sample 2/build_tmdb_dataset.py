# enhanced_tmdb_dataset_builder.py - Build Massive Dataset
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TMDB Configuration
TMDB_API_KEY = "4efced05afbb086b5fb03a39dcad3585"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

class MassiveDatasetBuilder:
    def __init__(self):
        self.all_movies = set()  # Track unique movie IDs
        self.failed_requests = 0
        self.successful_requests = 0
    
    def get_popular_movies(self, pages=500):  # Increased from 100
        """Fetch popular movies from TMDB"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), desc="📈 Fetching popular movies"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/movie/popular",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page,
                        "language": "en-US"
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                # Rate limiting
                time.sleep(0.05)  # Faster rate
                
            except Exception as e:
                self.failed_requests += 1
                if page <= 10:  # Only log errors for first 10 pages
                    logger.error(f"Error fetching popular page {page}: {str(e)}")
                continue
        
        return movies
    
    def get_top_rated_movies(self, pages=200):  # Increased from 50
        """Fetch top rated movies from TMDB"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), desc="⭐ Fetching top rated movies"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/movie/top_rated",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page,
                        "language": "en-US"
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                self.failed_requests += 1
                continue
        
        return movies
    
    def get_movies_by_genre(self, genre_id, genre_name, pages=100):  # Increased from 20
        """Fetch movies by specific genre"""
        movies = []
        
        # Get different sort orders for diversity
        sort_orders = [
            "popularity.desc",
            "vote_average.desc", 
            "release_date.desc",
            "revenue.desc",
            "vote_count.desc"
        ]
        
        pages_per_sort = max(1, pages // len(sort_orders))
        
        for sort_by in sort_orders:
            for page in tqdm(range(1, pages_per_sort + 1), 
                           desc=f"🎭 Genre {genre_name} ({sort_by})"):
                try:
                    response = requests.get(
                        f"{TMDB_BASE_URL}/discover/movie",
                        params={
                            "api_key": TMDB_API_KEY,
                            "with_genres": genre_id,
                            "page": page,
                            "language": "en-US",
                            "sort_by": sort_by,
                            "vote_count.gte": 10  # Minimum vote count for quality
                        }
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    movies.extend(data['results'])
                    self.successful_requests += 1
                    
                    time.sleep(0.05)
                    
                except Exception as e:
                    self.failed_requests += 1
                    continue
        
        return movies
    
    def get_movies_by_decade(self, start_year, end_year, pages=50):
        """Fetch movies from specific decade"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), 
                        desc=f"📅 Movies {start_year}-{end_year}"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/discover/movie",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page,
                        "language": "en-US",
                        "primary_release_date.gte": f"{start_year}-01-01",
                        "primary_release_date.lte": f"{end_year}-12-31",
                        "sort_by": "popularity.desc",
                        "vote_count.gte": 5
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                self.failed_requests += 1
                continue
        
        return movies
    
    def get_trending_movies(self, time_window="week", pages=10):
        """Fetch trending movies"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), desc=f"🔥 Trending ({time_window})"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/trending/movie/{time_window}",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                self.failed_requests += 1
                continue
        
        return movies
    
    def get_now_playing_movies(self, pages=10):
        """Fetch now playing movies"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), desc="🎬 Now playing"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/movie/now_playing",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page,
                        "language": "en-US"
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                self.failed_requests += 1
                continue
        
        return movies
    
    def get_upcoming_movies(self, pages=10):
        """Fetch upcoming movies"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), desc="🔮 Upcoming"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/movie/upcoming",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page,
                        "language": "en-US"
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                self.failed_requests += 1
                continue
        
        return movies
    
    def get_movies_by_rating_range(self, min_rating, max_rating, pages=30):
        """Fetch movies in specific rating range"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), 
                        desc=f"🌟 Rating {min_rating}-{max_rating}"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/discover/movie",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page,
                        "language": "en-US",
                        "vote_average.gte": min_rating,
                        "vote_average.lte": max_rating,
                        "sort_by": "popularity.desc",
                        "vote_count.gte": 20  # Ensure enough votes
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                self.failed_requests += 1
                continue
        
        return movies
    
    def get_movies_by_runtime(self, min_runtime, max_runtime, pages=20):
        """Fetch movies by runtime (short films, feature films, epics)"""
        movies = []
        
        for page in tqdm(range(1, pages + 1), 
                        desc=f"⏱️ Runtime {min_runtime}-{max_runtime} min"):
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/discover/movie",
                    params={
                        "api_key": TMDB_API_KEY,
                        "page": page,
                        "language": "en-US",
                        "with_runtime.gte": min_runtime,
                        "with_runtime.lte": max_runtime,
                        "sort_by": "popularity.desc"
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                movies.extend(data['results'])
                self.successful_requests += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                self.failed_requests += 1
                continue
        
        return movies
    
    def get_movie_details(self, movie_id):
        """Get detailed movie information including cast and crew"""
        try:
            response = requests.get(
                f"{TMDB_BASE_URL}/movie/{movie_id}",
                params={
                    "api_key": TMDB_API_KEY,
                    "append_to_response": "credits,keywords,similar,recommendations",
                    "language": "en-US"
                }
            )
            response.raise_for_status()
            
            movie_data = response.json()
            
            # Extract cast (top 10 actors for better matching)
            cast = []
            if 'credits' in movie_data and 'cast' in movie_data['credits']:
                cast = [actor['name'] for actor in movie_data['credits']['cast'][:10]]
            
            # Extract director and key crew
            director = None
            producer = None
            writer = None
            if 'credits' in movie_data and 'crew' in movie_data['credits']:
                for crew_member in movie_data['credits']['crew']:
                    if crew_member['job'] == 'Director' and not director:
                        director = crew_member['name']
                    elif crew_member['job'] == 'Producer' and not producer:
                        producer = crew_member['name']
                    elif crew_member['job'] in ['Writer', 'Screenplay'] and not writer:
                        writer = crew_member['name']
            
            # Extract genres
            genres = [genre['name'] for genre in movie_data.get('genres', [])]
            
            # Extract keywords (more for better matching)
            keywords = []
            if 'keywords' in movie_data and 'keywords' in movie_data['keywords']:
                keywords = [kw['name'] for kw in movie_data['keywords']['keywords'][:15]]
            
            # Extract production companies
            production_companies = []
            if 'production_companies' in movie_data:
                production_companies = [company['name'] for company in movie_data['production_companies'][:3]]
            
            return {
                'tmdb_id': movie_data['id'],
                'title': movie_data['title'],
                'overview': movie_data.get('overview', ''),
                'release_date': movie_data.get('release_date', ''),
                'vote_average': movie_data.get('vote_average', 0),
                'vote_count': movie_data.get('vote_count', 0),
                'poster_path': movie_data.get('poster_path', ''),
                'backdrop_path': movie_data.get('backdrop_path', ''),
                'genres': '|'.join(genres),
                'cast': '|'.join(cast),
                'director': director or '',
                'producer': producer or '',
                'writer': writer or '',
                'keywords': '|'.join(keywords),
                'production_companies': '|'.join(production_companies),
                'runtime': movie_data.get('runtime', 0),
                'budget': movie_data.get('budget', 0),
                'revenue': movie_data.get('revenue', 0),
                'original_language': movie_data.get('original_language', 'en'),
                'popularity': movie_data.get('popularity', 0),
                'adult': movie_data.get('adult', False)
            }
            
        except Exception as e:
            return None
    
    def get_genre_list(self):
        """Get all movie genres from TMDB"""
        try:
            response = requests.get(
                f"{TMDB_BASE_URL}/genre/movie/list",
                params={"api_key": TMDB_API_KEY}
            )
            response.raise_for_status()
            
            return response.json()['genres']
        except Exception as e:
            logger.error(f"Error fetching genres: {str(e)}")
            return []
    
    def build_massive_dataset(self):
        """Build a comprehensive movie dataset from TMDB - Target: 15,000-50,000 movies"""
        logger.info("🚀 Starting MASSIVE TMDB dataset creation...")
        logger.info("Target: 15,000+ unique movies with maximum diversity")
        
        print("=" * 80)
        print("🎬 BUILDING MASSIVE MOVIE DATASET")
        print("=" * 80)
        
        # 1. Popular movies (10,000+ movies)
        logger.info("1️⃣ Fetching popular movies...")
        popular_movies = self.get_popular_movies(pages=500)  # ~10,000 movies
        for movie in popular_movies:
            self.all_movies.add(movie['id'])
        print(f"   ✅ Popular movies: {len(popular_movies):,} collected, {len(self.all_movies):,} unique")
        
        # 2. Top rated movies (4,000+ movies)
        logger.info("2️⃣ Fetching top rated movies...")
        top_rated_movies = self.get_top_rated_movies(pages=200)
        for movie in top_rated_movies:
            self.all_movies.add(movie['id'])
        print(f"   ✅ Top rated movies: {len(top_rated_movies):,} collected, {len(self.all_movies):,} unique")
        
        # 3. Movies by genre (diverse collection)
        logger.info("3️⃣ Fetching movies by genre...")
        genres = self.get_genre_list()
        genre_count = 0
        for genre in genres:
            genre_movies = self.get_movies_by_genre(genre['id'], genre['name'], pages=100)
            for movie in genre_movies:
                self.all_movies.add(movie['id'])
            genre_count += len(genre_movies)
        print(f"   ✅ Genre movies: {genre_count:,} collected, {len(self.all_movies):,} unique")
        
        # 4. Movies by decade (historical diversity)
        logger.info("4️⃣ Fetching movies by decade...")
        decades = [
            (1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999),
            (2000, 2009), (2010, 2019), (2020, 2025)
        ]
        decade_count = 0
        for start_year, end_year in decades:
            decade_movies = self.get_movies_by_decade(start_year, end_year, pages=50)
            for movie in decade_movies:
                self.all_movies.add(movie['id'])
            decade_count += len(decade_movies)
        print(f"   ✅ Decade movies: {decade_count:,} collected, {len(self.all_movies):,} unique")
        
        # 5. Trending movies
        logger.info("5️⃣ Fetching trending movies...")
        trending_week = self.get_trending_movies("week", pages=10)
        trending_day = self.get_trending_movies("day", pages=10)
        for movie in trending_week + trending_day:
            self.all_movies.add(movie['id'])
        print(f"   ✅ Trending movies: {len(trending_week + trending_day):,} collected, {len(self.all_movies):,} unique")
        
        # 6. Now playing and upcoming
        logger.info("6️⃣ Fetching now playing and upcoming...")
        now_playing = self.get_now_playing_movies(pages=10)
        upcoming = self.get_upcoming_movies(pages=10)
        for movie in now_playing + upcoming:
            self.all_movies.add(movie['id'])
        print(f"   ✅ Current movies: {len(now_playing + upcoming):,} collected, {len(self.all_movies):,} unique")
        
        # 7. Movies by rating ranges (quality diversity)
        logger.info("7️⃣ Fetching movies by rating ranges...")
        rating_ranges = [
            (8.0, 10.0),  # Excellent movies
            (7.0, 7.9),   # Good movies
            (6.0, 6.9),   # Decent movies
            (4.0, 5.9),   # Below average (for completeness)
        ]
        rating_count = 0
        for min_rating, max_rating in rating_ranges:
            rating_movies = self.get_movies_by_rating_range(min_rating, max_rating, pages=30)
            for movie in rating_movies:
                self.all_movies.add(movie['id'])
            rating_count += len(rating_movies)
        print(f"   ✅ Rating range movies: {rating_count:,} collected, {len(self.all_movies):,} unique")
        
        # 8. Movies by runtime (length diversity)
        logger.info("8️⃣ Fetching movies by runtime...")
        runtime_ranges = [
            (60, 90),    # Short films
            (90, 120),   # Standard films
            (120, 180),  # Long films
            (180, 300)   # Epic films
        ]
        runtime_count = 0
        for min_runtime, max_runtime in runtime_ranges:
            runtime_movies = self.get_movies_by_runtime(min_runtime, max_runtime, pages=20)
            for movie in runtime_movies:
                self.all_movies.add(movie['id'])
            runtime_count += len(runtime_movies)
        print(f"   ✅ Runtime movies: {runtime_count:,} collected, {len(self.all_movies):,} unique")
        
        print("=" * 80)
        logger.info(f"📊 COLLECTION SUMMARY:")
        logger.info(f"   🎯 Total unique movies to process: {len(self.all_movies):,}")
        logger.info(f"   ✅ Successful API calls: {self.successful_requests:,}")
        logger.info(f"   ❌ Failed API calls: {self.failed_requests:,}")
        print("=" * 80)
        
        # Now get detailed information for each movie
        logger.info("9️⃣ Fetching detailed movie information...")
        detailed_movies = []
        failed_movies = []
        
        # Process in chunks with progress tracking
        movie_list = list(self.all_movies)
        chunk_size = 100
        
        for i in tqdm(range(0, len(movie_list), chunk_size), desc="🎬 Processing movie details"):
            chunk = movie_list[i:i + chunk_size]
            
            for movie_id in chunk:
                movie_details = self.get_movie_details(movie_id)
                if movie_details:
                    # Add year from release_date
                    if movie_details['release_date']:
                        try:
                            movie_details['year'] = int(movie_details['release_date'][:4])
                        except:
                            movie_details['year'] = None
                    else:
                        movie_details['year'] = None
                    
                    # Add full poster and backdrop URLs
                    if movie_details['poster_path']:
                        movie_details['poster_url'] = f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
                    else:
                        movie_details['poster_url'] = None
                    
                    if movie_details['backdrop_path']:
                        movie_details['backdrop_url'] = f"https://image.tmdb.org/t/p/w1280{movie_details['backdrop_path']}"
                    else:
                        movie_details['backdrop_url'] = None
                    
                    detailed_movies.append(movie_details)
                else:
                    failed_movies.append(movie_id)
                
                # Rate limiting
                time.sleep(0.05)
            
            # Progress update every chunk
            if i % (chunk_size * 10) == 0:
                logger.info(f"   Processed {i:,}/{len(movie_list):,} movies...")
        
        logger.info(f"✅ Successfully processed: {len(detailed_movies):,} movies")
        logger.info(f"❌ Failed to process: {len(failed_movies):,} movies")
        
        # Create DataFrame and clean data
        logger.info("🧹 Cleaning and filtering dataset...")
        df = pd.DataFrame(detailed_movies)
        
        # Filter out movies without essential information
        original_size = len(df)
        df = df[
            (df['overview'].notna()) & 
            (df['overview'] != '') & 
            (df['overview'].str.len() > 20) &  # Meaningful overview
            (df['title'].notna()) & 
            (df['genres'] != '') &
            (df['vote_count'] >= 5) &  # Minimum vote count
            (~df['adult']) &  # No adult content
            (df['runtime'] > 0) &  # Valid runtime
            (df['year'].notna()) &
            (df['year'] >= 1960) &  # Modern cinema
            (df['year'] <= 2025)
        ]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title', 'year'])
        
        # Sort by popularity and vote count
        df = df.sort_values(['popularity', 'vote_count'], ascending=[False, False])
        
        logger.info(f"🧹 Filtered from {original_size:,} to {len(df):,} high-quality movies")
        
        # Save to CSV
        output_path = "./Dataset/tmdb_movies_massive.csv"
        df.to_csv(output_path, index=False)
        
        # Generate statistics
        logger.info("📊 Generating dataset statistics...")
        
        print("\n" + "=" * 80)
        print("📊 FINAL DATASET STATISTICS")
        print("=" * 80)
        print(f"📁 Dataset saved to: {output_path}")
        print(f"📊 Total movies: {len(df):,}")
        print(f"⭐ Average rating: {df['vote_average'].mean():.2f}")
        print(f"🗳️  Average vote count: {df['vote_count'].mean():.0f}")
        print(f"🎭 Total unique genres: {len(set('|'.join(df['genres'].fillna('')).split('|')))}")
        print(f"🎬 Movies with cast info: {len(df[df['cast'] != '']):,} ({len(df[df['cast'] != '']) / len(df) * 100:.1f}%)")
        print(f"🎪 Movies with director info: {len(df[df['director'] != '']):,} ({len(df[df['director'] != '']) / len(df) * 100:.1f}%)")
        print(f"🏷️ Movies with keywords: {len(df[df['keywords'] != '']):,} ({len(df[df['keywords'] != '']) / len(df) * 100:.1f}%)")
        print(f"🏢 Movies with production companies: {len(df[df['production_companies'] != '']):,}")
        print(f"📅 Year range: {int(df['year'].min())} - {int(df['year'].max())}")
        print(f"⏱️ Runtime range: {int(df['runtime'].min())} - {int(df['runtime'].max())} minutes")
        print(f"💰 Budget range: ${df['budget'].min():,.0f} - ${df['budget'].max():,.0f}")
        
        # Top genres
        all_genres = []
        for genres in df['genres'].dropna():
            all_genres.extend(genres.split('|'))
        genre_counts = pd.Series(all_genres).value_counts()
        print(f"\n🔥 Top 10 genres:")
        for genre, count in genre_counts.head(10).items():
            print(f"   {genre}: {count:,} movies")
        
        # Movies by decade
        decade_counts = df['year'].apply(lambda x: f"{int(x//10)*10}s").value_counts().sort_index()
        print(f"\n📅 Movies by decade:")
        for decade, count in decade_counts.items():
            print(f"   {decade}: {count:,} movies")
        
        print("=" * 80)
        
        return df

if __name__ == "__main__":
    # Create Dataset directory if it doesn't exist
    os.makedirs("./Dataset", exist_ok=True)
    
    # Build the massive dataset
    builder = MassiveDatasetBuilder()
    dataset = builder.build_massive_dataset()
    
    print("\n🎉 MASSIVE TMDB Dataset created successfully!")
    print(f"📈 Dataset size increased from ~3,000 to {len(dataset):,} movies")
    print(f"📁 Saved to: ./Dataset/tmdb_movies_massive.csv")
    print(f"💾 Estimated file size: ~{len(dataset) * 2.6 / 3000:.1f}MB")
    print("\n🚀 Your recommendation model will now have much better accuracy!")
    print("   Update your API to use 'tmdb_movies_massive.csv' instead of 'tmdb_movies.csv'")
    
    # Show file size
    if os.path.exists("./Dataset/tmdb_movies_massive.csv"):
        file_size = os.path.getsize("./Dataset/tmdb_movies_massive.csv") / (1024 * 1024)
        print(f"💾 Actual file size: {file_size:.1f} MB")