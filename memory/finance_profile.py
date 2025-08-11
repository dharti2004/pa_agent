from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class FinanceProfile(BaseModel):
    """Comprehensive financial profile with clearly named fields for easy data storage and retrieval."""
    
    # Basic Info - "Tell me about yourself"
    user_age_years: Optional[int] = None
    user_occupation_job: Optional[str] = None
    user_location_city_country: Optional[str] = None
    user_has_spouse_partner: Optional[bool] = None
    user_number_of_children: Optional[int] = None
    
    # Income - "How much do you earn?"
    income_total_yearly_amount: Optional[Decimal] = None
    income_monthly_take_home: Optional[Decimal] = None
    income_from_salary: Optional[Decimal] = None
    income_from_business: Optional[Decimal] = None
    income_from_investments: Optional[Decimal] = None
    income_from_rental: Optional[Decimal] = None
    income_other_sources: Optional[str] = None
    
    # Expenses - "What are your monthly expenses?"
    expense_total_monthly: Optional[Decimal] = None
    expense_housing_rent_mortgage: Optional[Decimal] = None
    expense_food_groceries_dining: Optional[Decimal] = None
    expense_transportation_car_gas: Optional[Decimal] = None
    expense_utilities_bills: Optional[Decimal] = None
    expense_entertainment_hobbies: Optional[Decimal] = None
    expense_other_categories: Optional[str] = None
    
    # Savings - "How much do you save?"
    savings_monthly_amount: Optional[Decimal] = None
    savings_emergency_fund_total: Optional[Decimal] = None
    savings_percentage_of_income: Optional[float] = None
    
    # Goals - "What are your financial goals?"
    goal_buy_house_amount_needed: Optional[Decimal] = None
    goal_buy_house_timeline_years: Optional[int] = None
    goal_retirement_target_amount: Optional[Decimal] = None
    goal_retirement_age_target: Optional[int] = None
    goal_children_education_amount: Optional[Decimal] = None
    goal_vacation_travel_budget: Optional[Decimal] = None
    goal_other_major_purchases: Optional[str] = None
    goal_most_important_priority: Optional[str] = None
    
    # Current Investments - "What investments do you have?"
    investment_total_portfolio_value: Optional[Decimal] = None
    investment_stocks_value: Optional[Decimal] = None
    investment_bonds_value: Optional[Decimal] = None
    investment_mutual_funds_value: Optional[Decimal] = None
    investment_etf_value: Optional[Decimal] = None
    investment_crypto_value: Optional[Decimal] = None
    investment_real_estate_value: Optional[Decimal] = None
    investment_gold_precious_metals: Optional[Decimal] = None
    investment_fixed_deposits_value: Optional[Decimal] = None
    investment_retirement_401k_ira: Optional[Decimal] = None
    
    # Debt - "What debts do you have?"
    debt_total_amount_owed: Optional[Decimal] = None
    debt_home_loan_mortgage: Optional[Decimal] = None
    debt_car_loan_amount: Optional[Decimal] = None
    debt_student_loan_amount: Optional[Decimal] = None
    debt_credit_card_balance: Optional[Decimal] = None
    debt_personal_loan_amount: Optional[Decimal] = None
    debt_monthly_total_payments: Optional[Decimal] = None
    
    # Risk Profile - "What's your risk tolerance?"
    risk_tolerance_level: Optional[str] = None  # "low", "medium", "high"
    risk_can_afford_to_lose_amount: Optional[Decimal] = None
    risk_investment_experience_years: Optional[int] = None
    risk_reaction_to_market_drop: Optional[str] = None  # "sell", "hold", "buy more"
    
    # Investment Preferences - "How do you like to invest?"
    prefer_stocks_over_bonds: Optional[bool] = None
    prefer_domestic_over_international: Optional[bool] = None
    prefer_growth_over_dividend: Optional[bool] = None
    prefer_individual_stocks_over_funds: Optional[bool] = None
    prefer_active_over_passive: Optional[bool] = None
    interested_in_crypto: Optional[bool] = None
    interested_in_real_estate: Optional[bool] = None
    avoid_specific_sectors: Optional[List[str]] = None
    
    # Tax Situation - "What's your tax situation?"
    tax_bracket_percentage: Optional[float] = None
    tax_filing_status: Optional[str] = None  # "single", "married", etc.
    tax_state_residence: Optional[str] = None
    tax_deductions_claimed: Optional[List[str]] = None
    
    # Insurance - "What insurance do you have?"
    insurance_has_life: Optional[bool] = None
    insurance_has_health: Optional[bool] = None
    insurance_has_disability: Optional[bool] = None
    insurance_has_home_renters: Optional[bool] = None
    insurance_has_auto: Optional[bool] = None
    
    # Financial Knowledge - "How much do you know about investing?"
    knowledge_understands_stocks: Optional[bool] = None
    knowledge_understands_bonds: Optional[bool] = None
    knowledge_understands_mutual_funds: Optional[bool] = None
    knowledge_needs_basic_education: Optional[bool] = None
    knowledge_confident_level: Optional[str] = None  # "beginner", "intermediate", "advanced"
    
    # Past Behavior - "Tell me about your investment history"
    history_best_investment_made: Optional[str] = None
    history_worst_investment_made: Optional[str] = None
    history_learned_lessons: Optional[str] = None
    history_years_investing: Optional[int] = None
    
    # Future Plans - "What are your future plans?"
    plans_major_purchase_next_year: Optional[str] = None
    plans_career_change_expected: Optional[bool] = None
    plans_expecting_inheritance: Optional[bool] = None
    plans_starting_business: Optional[bool] = None
    
    # Preferences - "How do you want advice?"
    prefers_simple_explanations: Optional[bool] = None
    prefers_detailed_analysis: Optional[bool] = None
    prefers_conservative_advice: Optional[bool] = None
    prefers_aggressive_strategies: Optional[bool] = None
    
    # Tracking - "When did we last update?"
    profile_last_updated_date: Optional[datetime] = None
    portfolio_last_reviewed_date: Optional[datetime] = None
    goals_last_discussed_date: Optional[datetime] = None
    
    # Open Notes - "Anything else to remember?"
    notes_special_circumstances: Optional[str] = None
    notes_specific_questions_asked: Optional[List[str]] = None
    notes_advice_given_previously: Optional[List[str]] = None