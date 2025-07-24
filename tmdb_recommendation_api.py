import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import logging
from dotenv import load_dotenv
import requests
import time
from datetime import datetime
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TMDB Movie Recommendation API", 
    version="2.0.0",
    description="Movie recommendations using TMDB dataset with genre, overview, cast, and director analysis"
)

# Add CORS middleware - UPDATE THIS WITH YOUR PARTNER'S DOMAIN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your partner's domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables
movies_df = None
tfidf_vectorizer = None
tfidf_matrix = None
genre_vectorizer = None
genre_matrix = None
cast_vectorizer = None
cast_matrix = None
director_vectorizer = None
director_matrix = None

# TMDB Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "4efced05afbb086b5fb03a39dcad3585")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Request/Response Models
class MovieRequest(BaseModel):
    title: str = Field(..., description="Movie title")
    n_recommendations: Optional[int] = Field(10, ge=1, le=20, description="Number of recommendations")

class MovieRecommendation(BaseModel):
    title: str
    year: Optional[int] = None
    poster_url: Optional[str] = None
    rating: float
    similarity_score: float

class RecommendationResponse(BaseModel):
    query_movie: Dict
    recommendations: List[MovieRecommendation]
    total_found: int

def build_dataset_on_server():
    """Build dataset on server startup if it doesn't exist"""
    dataset_path = "./Dataset/tmdb_movies_massive.csv"
    
    # Create Dataset directory if it doesn't exist
    os.makedirs("./Dataset", exist_ok=True)
    
    if os.path.exists(dataset_path):
        logger.info("Dataset already exists, skipping build")
        return dataset_path
    
    logger.info("Building dataset on server...")
    
    try:
        # Import the dataset building functions
        from build_tmdb_dataset import build_comprehensive_dataset
        
        # Build the dataset
        df = build_comprehensive_dataset()
        logger.info(f"Dataset built successfully with {len(df)} movies")
        
        return dataset_path
        
    except Exception as e:
        logger.error(f"Failed to build dataset: {str(e)}")
        # Return None to indicate failure
        return None

def load_tmdb_dataset(dataset_path: str = "./Dataset/tmdb_movies.csv"):
    """Load TMDB dataset and prepare all vectorizers"""
    global movies_df, tfidf_vectorizer, tfidf_matrix, genre_vectorizer, genre_matrix
    global cast_vectorizer, cast_matrix, director_vectorizer, director_matrix
    
    try:
        # Try to build dataset if it doesn't exist
        if not os.path.exists(dataset_path):
            logger.info("Dataset not found, building on server...")
            dataset_path = build_dataset_on_server()
            if dataset_path is None:
                raise Exception("Failed to build dataset on server")
        
        logger.info("Loading TMDB dataset...")
        
        # Load the dataset
        movies_df = pd.read_csv(dataset_path)
        
        # Clean and prepare data
        movies_df['overview'] = movies_df['overview'].fillna('')
        movies_df['genres'] = movies_df['genres'].fillna('')
        movies_df['cast'] = movies_df['cast'].fillna('')
        movies_df['director'] = movies_df['director'].fillna('')
        
        # Remove movies without essential information
        movies_df = movies_df[
            (movies_df['overview'] != '') & 
            (movies_df['genres'] != '') &
            (movies_df['title'].notna())
        ]
        
        logger.info(f"Loaded {len(movies_df)} movies")
        
        # 1. Overview TF-IDF Vectorizer (Primary)
        logger.info("Creating overview TF-IDF matrix...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['overview'])
        
        # 2. Genre TF-IDF Vectorizer (Primary)
        logger.info("Creating genre TF-IDF matrix...")
        genre_texts = movies_df['genres'].str.replace('|', ' ')
        genre_vectorizer = TfidfVectorizer(
            max_features=100,
            token_pattern=r'\b[A-Za-z][A-Za-z\s]*[A-Za-z]\b',
            lowercase=True
        )
        genre_matrix = genre_vectorizer.fit_transform(genre_texts)
        
        # 3. Cast TF-IDF Vectorizer (Secondary)
        logger.info("Creating cast TF-IDF matrix...")
        cast_texts = movies_df['cast'].str.replace('|', ' ')
        cast_vectorizer = TfidfVectorizer(
            max_features=2000,
            token_pattern=r'\b[A-Za-z][A-Za-z\s]*[A-Za-z]\b',
            lowercase=True,
            min_df=2
        )
        cast_matrix = cast_vectorizer.fit_transform(cast_texts)
        
        # 4. Director TF-IDF Vectorizer (Secondary)
        logger.info("Creating director TF-IDF matrix...")
        director_texts = movies_df['director'].fillna('')
        director_vectorizer = TfidfVectorizer(
            max_features=500,
            token_pattern=r'\b[A-Za-z][A-Za-z\s]*[A-Za-z]\b',
            lowercase=True,
            min_df=2
        )
        director_matrix = director_vectorizer.fit_transform(director_texts)
        
        logger.info("All vectorizers ready!")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def find_movie_by_title(title: str):
    """Find movie by title with fuzzy matching"""
    title_clean = title.lower().strip()
    
    # Try exact match first
    exact_match = movies_df[movies_df['title'].str.lower() == title_clean]
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    # Try partial match
    partial_match = movies_df[movies_df['title'].str.lower().str.contains(title_clean, na=False)]
    if not partial_match.empty:
        return partial_match.loc[partial_match['vote_count'].idxmax()]
    
    return None

def get_movie_recommendations(movie_title: str, n_recommendations: int = 10):
    """Get recommendations using combined similarity scores"""
    
    # Find the query movie
    query_movie = find_movie_by_title(movie_title)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found")
    
    query_idx = query_movie.name
    
    # Calculate similarity scores for each component with better weights
    similarities = {}
    
    # 1. Overview similarity (45% weight - more important for plot)
    overview_similarities = cosine_similarity(tfidf_matrix[query_idx], tfidf_matrix).flatten()
    similarities['overview'] = overview_similarities * 0.45
    
    # 2. Genre similarity (30% weight)
    genre_similarities = cosine_similarity(genre_matrix[query_idx], genre_matrix).flatten()
    similarities['genre'] = genre_similarities * 0.30
    
    # 3. Cast similarity (15% weight)
    cast_similarities = cosine_similarity(cast_matrix[query_idx], cast_matrix).flatten()
    similarities['cast'] = cast_similarities * 0.15
    
    # 4. Director similarity (10% weight)
    director_similarities = cosine_similarity(director_matrix[query_idx], director_matrix).flatten()
    similarities['director'] = director_similarities * 0.10
    
    # Combine all similarities
    combined_similarities = (
        similarities['overview'] + 
        similarities['genre'] + 
        similarities['cast'] + 
        similarities['director']
    )
    
    # Get top similar movies (excluding the query movie itself)
    similar_indices = combined_similarities.argsort()[::-1]
    
    recommendations = []
    seen_exact_titles = set()
    query_title_clean = query_movie['title'].lower().strip()
    seen_exact_titles.add(query_title_clean)
    
    for idx in similar_indices:
        if len(recommendations) >= n_recommendations:
            break
        
        if idx == query_idx:
            continue
            
        movie = movies_df.iloc[idx]
        movie_title_clean = movie['title'].lower().strip()
        
        # Only avoid EXACT duplicates (allow sequels/prequels)
        if movie_title_clean in seen_exact_titles:
            continue
        
        seen_exact_titles.add(movie_title_clean)
        
        # Lower similarity threshold to include more good matches
        if combined_similarities[idx] < 0.005:  # Much lower threshold
            continue
        
        # Bonus for high ratings (boost similarity score for highly rated movies)
        rating_bonus = 0
        if pd.notna(movie['vote_average']) and movie['vote_average'] >= 7.0:
            rating_bonus = 0.05
        elif pd.notna(movie['vote_average']) and movie['vote_average'] >= 8.0:
            rating_bonus = 0.10
        
        final_similarity = combined_similarities[idx] + rating_bonus
        
        # Create recommendation
        recommendation = MovieRecommendation(
            title=movie['title'],
            year=int(movie['year']) if pd.notna(movie['year']) else None,
            poster_url=movie['poster_url'] if pd.notna(movie['poster_url']) else None,
            rating=float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0.0,
            similarity_score=float(final_similarity)  # Keep for sorting, but won't show in frontend
        )
        
        recommendations.append(recommendation)
    
    # Sort by rating (highest first) for better recommendations
    recommendations.sort(key=lambda x: x.rating, reverse=True)
    
    # Query movie info for response
    query_movie_info = {
        'title': query_movie['title'],
        'year': int(query_movie['year']) if pd.notna(query_movie['year']) else None,
        'poster_url': query_movie['poster_url'] if pd.notna(query_movie['poster_url']) else None,
        'rating': float(query_movie['vote_average']) if pd.notna(query_movie['vote_average']) else 0.0,
        'genres': query_movie['genres']
    }
    
    return RecommendationResponse(
        query_movie=query_movie_info,
        recommendations=recommendations,
        total_found=len(recommendations)
    )

@app.on_event("startup")
async def startup_event():
    """Load dataset when API starts"""
    try:
        load_tmdb_dataset()
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "TMDB Movie Recommendation API is running!",
        "total_movies": len(movies_df) if movies_df is not None else 0,
        "dataset_source": "TMDB Custom Dataset",
        "version": "2.0.0"
    }

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(movie_request: MovieRequest):
    """Get movie recommendations based on title"""
    try:
        return get_movie_recommendations(
            movie_request.title,
            movie_request.n_recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{movie_title}", tags=["Search"])
async def search_movie(movie_title: str):
    """Search for a movie in the dataset"""
    try:
        movie = find_movie_by_title(movie_title)
        if movie is None:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        return {
            "title": movie['title'],
            "year": int(movie['year']) if pd.notna(movie['year']) else None,
            "poster_url": movie['poster_url'] if pd.notna(movie['poster_url']) else None,
            "rating": float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0.0,
            "genres": movie['genres'],
            "overview": movie['overview']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_movie: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Statistics"])
async def get_dataset_stats():
    """Get statistics about the dataset"""
    if movies_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    try:
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            all_genres.update(genres.split('|'))
        
        stats = {
            "total_movies": len(movies_df),
            "average_rating": float(movies_df['vote_average'].mean()),
            "total_genres": len(all_genres),
            "year_range": {
                "min": int(movies_df['year'].min()),
                "max": int(movies_df['year'].max())
            },
            "movies_with_posters": len(movies_df[movies_df['poster_url'].notna()]),
            "average_vote_count": float(movies_df['vote_count'].mean())
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/popular", tags=["Browse"])
async def get_popular_movies(limit: int = Query(20, ge=1, le=50)):
    """Get popular movies from the dataset"""
    try:
        popular = movies_df.nlargest(limit, 'vote_count')
        
        results = []
        for _, movie in popular.iterrows():
            results.append({
                "title": movie['title'],
                "year": int(movie['year']) if pd.notna(movie['year']) else None,
                "poster_url": movie['poster_url'] if pd.notna(movie['poster_url']) else None,
                "rating": float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0.0
            })
        
        return {
            "popular_movies": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in get_popular_movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "tmdb_recommendation_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )