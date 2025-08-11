from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class TravelProfile(BaseModel):
    # ===== Basic Preferences =====
    travel_style: Optional[str] = None  # "budget", "luxury", etc.
    accommodation_type: Optional[List[str]] = None  # ["hotels", "airbnb"]
    group_size_preference: Optional[str] = None  # "solo", "couple", etc.
    travel_pace: Optional[str] = None  # "fast", "moderate", "slow"
    planning_style: Optional[str] = None  # "detailed", "spontaneous", etc.

    # ===== Travel History =====
    visited_destinations: Optional[List[Dict[str, str]]] = None  # [{"country": "Japan", "city": "Tokyo", "date": "2023-05"}]
    favorite_destinations: Optional[List[str]] = None
    disappointing_destinations: Optional[List[str]] = None
    revisit_list: Optional[List[str]] = None
    return_to_destinations: Optional[List[str]] = None
    trip_frequency: Optional[str] = None  # "monthly", etc.
    typical_trip_duration: Optional[str] = None  # "weekend", "2_weeks"
    typical_vacation_length_days: Optional[int] = None
    medical_conditions_to_account: Optional[List[str]] = None  # ["asthma", "low altitude only"]
    needs_digital_connectivity: Optional[bool] = None  # "Yes" if they need Wi-Fi or remote work tools


    # ===== Interests and Activities =====
    preferred_activities: Optional[List[str]] = None  # ["hiking", "beaches"]
    cultural_interests: Optional[List[str]] = None  # ["art", "festivals"]
    special_interests: Optional[List[str]] = None  # ["photography", "diving"]
    activity_level: Optional[str] = None  # "moderate", "active", etc.

    # ===== Practical & Budget Constraints =====
    budget_range: Optional[str] = None  # "moderate", "luxury", etc.
    budget_per_day_usd: Optional[int] = None
    spending_categories: Optional[Dict[str, float]] = None  # {"food": 100, "shopping": 200}
    dietary_restrictions: Optional[List[str]] = None  # ["vegetarian", "halal"]
    mobility_constraints: Optional[str] = None  # "moderate", "significant"

    # ===== Transportation Preferences =====
    preferred_transport_modes: Optional[List[str]] = None  # ["train", "plane"]
    flight_preferences: Optional[Dict[str, str]] = None  # {"class": "economy", "type": "direct"}
    preferred_airlines: Optional[List[str]] = None
    local_transport_options: Optional[List[str]] = None  # ["rental_car", "tours"]
    max_flight_duration_hours: Optional[int] = None  # e.g. 8

    # ===== Environment & Climate =====
    climate_preferences: Optional[List[str]] = None  # ["cold", "tropical"]
    preferred_seasons: Optional[List[str]] = None  # ["summer", "spring"]
    geography_preferences: Optional[List[str]] = None  # ["beach", "mountain"]
    crowd_tolerance: Optional[str] = None  # "prefers_quiet", "loves_crowds", etc.

    # ===== Social & Cultural Preferences =====
    cultural_adventurousness: Optional[str] = None  # "moderate", "familiar_only"
    food_adventurousness: Optional[str] = None  # "very_adventurous", etc.
    language_comfort_level: Optional[str] = None  # "only_english", "multilingual"
    social_preference: Optional[str] = None  # "prefers_privacy", etc.

    # ===== Trip Intent / Purpose =====
    typical_travel_companions: Optional[List[str]] = None  # ["spouse", "family"]
    trip_purposes: Optional[List[str]] = None  # ["relaxation", "business"]
    special_occasions: Optional[List[str]] = None  # ["birthday", "honeymoon"]

    # ===== Booking Behavior =====
    booking_advance_notice: Optional[str] = None  # "last_minute", "months"
    price_sensitivity_level: Optional[str] = None  # "very_sensitive", etc.
    preferred_booking_platforms: Optional[List[str]] = None  # ["booking.com", "direct"]

    # ===== Personality & Behavior Traits =====
    travel_experience_level: Optional[str] = None  # "beginner", "expert"
    revisits_preference: Optional[str] = None  # "loves_new_places", etc.
    trip_spontaneity_level: Optional[str] = None  # "structured", "spontaneous"
    daily_rhythm: Optional[str] = None  # "early_riser", "night_owl", "neutral"

    # ===== Destination Intentions =====
    bucket_list_destinations: Optional[List[str]] = None
    avoid_regions: Optional[List[str]] = None
    preferred_continents: Optional[List[str]] = None

    # ===== Availability / Calendar =====
    travel_blackout_dates: Optional[List[str]] = None  # ["2025-01-01"]
    prefers_long_weekends: Optional[bool] = None
    available_holidays_per_year: Optional[int] = None

    # ===== Personalization Behavior =====
    exploration_vs_familiarity: Optional[str] = None  # "explore", "repeat_favorites"
    surprise_trip_tolerance: Optional[str] = None  # "loves_surprises", "needs_plan"
    language_assistance_required: Optional[bool] = None
    trip_customization_preference: Optional[str] = None  # "custom", "prebuilt"

    # ===== Sharing & Collaboration =====
    shareable_profile_enabled: Optional[bool] = None
    collaborative_trip_planning_allowed: Optional[bool] = None

    # ===== AI Insights / Enhancements (Optional) =====
    inferred_travel_persona: Optional[str] = None  # e.g., "Explorer", "Relaxer"
    ai_generated_profile_summary: Optional[str] = None
    trip_feedback_history: Optional[List[Dict[str, str]]] = None  # [{"trip": "Paris", "rating": "positive"}]

    # ===== Legal & Geo Data =====
    home_country: Optional[str] = None
    passport_country: Optional[str] = None
    visa_free_countries: Optional[List[str]] = None
    visa_restricted_countries: Optional[List[str]] = None
    disliked_regions: Optional[List[str]] = None

    # ===== System / Metadata =====
    profile_last_updated: Optional[datetime] = None
    profile_completeness_score: Optional[float] = None  # e.g. 0.85 meaning 85% filled