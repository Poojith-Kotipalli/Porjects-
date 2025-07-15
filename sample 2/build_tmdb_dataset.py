# build_tmdb_dataset.py
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TMDB Configuration
TMDB_API_KEY = "4efced05afbb086b5fb03a39dcad3585"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_popular_movies(pages=50):
    """Fetch popular movies from TMDB"""
    movies = []
    
    for page in tqdm(range(1, pages + 1), desc="Fetching popular movies"):
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
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching page {page}: {str(e)}")
            continue
    
    return movies

def get_top_rated_movies(pages=30):
    """Fetch top rated movies from TMDB"""
    movies = []
    
    for page in tqdm(range(1, pages + 1), desc="Fetching top rated movies"):
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
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching page {page}: {str(e)}")
            continue
    
    return movies

def get_movies_by_genre(genre_id, pages=20):
    """Fetch movies by specific genre"""
    movies = []
    
    for page in tqdm(range(1, pages + 1), desc=f"Fetching movies for genre {genre_id}"):
        try:
            response = requests.get(
                f"{TMDB_BASE_URL}/discover/movie",
                params={
                    "api_key": TMDB_API_KEY,
                    "with_genres": genre_id,
                    "page": page,
                    "language": "en-US",
                    "sort_by": "popularity.desc"
                }
            )
            response.raise_for_status()
            
            data = response.json()
            movies.extend(data['results'])
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching page {page} for genre {genre_id}: {str(e)}")
            continue
    
    return movies

def get_movie_details(movie_id):
    """Get detailed movie information including cast and crew"""
    try:
        # Get movie details
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{movie_id}",
            params={
                "api_key": TMDB_API_KEY,
                "append_to_response": "credits,keywords",
                "language": "en-US"
            }
        )
        response.raise_for_status()
        
        movie_data = response.json()
        
        # Extract cast (top 5 actors)
        cast = []
        if 'credits' in movie_data and 'cast' in movie_data['credits']:
            cast = [actor['name'] for actor in movie_data['credits']['cast'][:5]]
        
        # Extract director
        director = None
        if 'credits' in movie_data and 'crew' in movie_data['credits']:
            for crew_member in movie_data['credits']['crew']:
                if crew_member['job'] == 'Director':
                    director = crew_member['name']
                    break
        
        # Extract genres
        genres = [genre['name'] for genre in movie_data.get('genres', [])]
        
        # Extract keywords
        keywords = []
        if 'keywords' in movie_data and 'keywords' in movie_data['keywords']:
            keywords = [kw['name'] for kw in movie_data['keywords']['keywords'][:10]]
        
        return {
            'tmdb_id': movie_data['id'],
            'title': movie_data['title'],
            'overview': movie_data.get('overview', ''),
            'release_date': movie_data.get('release_date', ''),
            'vote_average': movie_data.get('vote_average', 0),
            'vote_count': movie_data.get('vote_count', 0),
            'poster_path': movie_data.get('poster_path', ''),
            'genres': '|'.join(genres),
            'cast': '|'.join(cast),
            'director': director or '',
            'keywords': '|'.join(keywords),
            'runtime': movie_data.get('runtime', 0),
            'budget': movie_data.get('budget', 0),
            'revenue': movie_data.get('revenue', 0)
        }
        
    except Exception as e:
        logger.error(f"Error fetching details for movie {movie_id}: {str(e)}")
        return None

def get_genre_list():
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

def build_comprehensive_dataset():
    """Build a comprehensive movie dataset from TMDB"""
    logger.info("Starting TMDB dataset creation...")
    
    # Get all unique movies from different sources
    all_movies = set()
    
    # 1. Popular movies
    logger.info("Fetching popular movies...")
    popular_movies = get_popular_movies(pages=100)  # ~2000 movies
    for movie in popular_movies:
        all_movies.add(movie['id'])
    
    # 2. Top rated movies
    logger.info("Fetching top rated movies...")
    top_rated_movies = get_top_rated_movies(pages=50)  # ~1000 movies
    for movie in top_rated_movies:
        all_movies.add(movie['id'])
    
    # 3. Movies by genre to ensure diversity
    logger.info("Fetching movies by genre...")
    genres = get_genre_list()
    for genre in genres:
        genre_movies = get_movies_by_genre(genre['id'], pages=10)  # ~200 per genre
        for movie in genre_movies:
            all_movies.add(movie['id'])
    
    logger.info(f"Total unique movies to process: {len(all_movies)}")
    
    # Now get detailed information for each movie
    detailed_movies = []
    failed_movies = []
    
    for movie_id in tqdm(all_movies, desc="Getting movie details"):
        movie_details = get_movie_details(movie_id)
        if movie_details:
            # Add year from release_date
            if movie_details['release_date']:
                try:
                    movie_details['year'] = int(movie_details['release_date'][:4])
                except:
                    movie_details['year'] = None
            else:
                movie_details['year'] = None
            
            # Add full poster URL
            if movie_details['poster_path']:
                movie_details['poster_url'] = f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
            else:
                movie_details['poster_url'] = None
            
            detailed_movies.append(movie_details)
        else:
            failed_movies.append(movie_id)
        
        # Rate limiting
        time.sleep(0.1)
    
    logger.info(f"Successfully processed: {len(detailed_movies)} movies")
    logger.info(f"Failed to process: {len(failed_movies)} movies")
    
    # Create DataFrame and save
    df = pd.DataFrame(detailed_movies)
    
    # Filter out movies without overview or essential information
    df = df[
        (df['overview'].notna()) & 
        (df['overview'] != '') & 
        (df['title'].notna()) & 
        (df['genres'] != '')
    ]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title', 'year'])
    
    # Save to CSV
    output_path = "./Dataset/tmdb_movies.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Final dataset size: {len(df)} movies")
    
    # Print some statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Average rating: {df['vote_average'].mean():.2f}")
    logger.info(f"Total genres: {len(set('|'.join(df['genres'].fillna('')).split('|')))}")
    logger.info(f"Movies with cast info: {len(df[df['cast'] != ''])}")
    logger.info(f"Movies with director info: {len(df[df['director'] != ''])}")
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    return df

if __name__ == "__main__":
    # Create Dataset directory if it doesn't exist
    os.makedirs("./Dataset", exist_ok=True)
    
    # Build the dataset
    dataset = build_comprehensive_dataset()
    
    print("\n✅ TMDB Dataset created successfully!")
    print(f"📁 Saved to: ./Dataset/tmdb_movies.csv")
    print(f"📊 Total movies: {len(dataset)}")
    print("\n🚀 You can now run the recommendation API with this dataset!")