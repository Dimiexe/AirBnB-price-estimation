import pandas as pd

def engineer_bathroom_text(df_input):
    ## Turning bathroom_texts into 2 more useful features, the actual number of bathrooms and if it is shared or not.
    df = df_input.copy()
    
    if "bathrooms_text" in df.columns:
        # Fill NaN values with 0, assuming the listing has no bath.
        df.loc[:,"bathrooms_text"].fillna(value = '0 baths', inplace = True)
        # Create two new columns.
        bathrooms = []                  # 'bathrooms' will hold the amount of bathrooms the listing has. 
        shared_bath = []                # 'shared_bath' for if the bathrooms are shared.
        
        for text in df.loc[:,"bathrooms_text"]:
            text = text.lower()
        
            if 'shared' in text:
                shared_bath.append(1)
            else:
                shared_bath.append(0)

            half_flag = False
            if 'half-bath' in text:
                half_flag = True
        
            text = text.split()
        
            try:
                baths = float(text[0])
                if half_flag:
                    baths += 0.5
                bathrooms.append(baths)        
            except:
                bathrooms.append(0)
        
        df.loc[:,'bathrooms'] = bathrooms
        df.loc[:,'shared_bath'] = shared_bath
        df.drop("bathrooms_text", axis=1,inplace=True)
    else:
        print("No feature 'bathrooms_text' was found. Returning unaffected DataFrame.")
    return df

def process_price(x):
    ## Clean the "price" feature of the dataset.
    temp = x.split('$')[1]
    temp = temp.replace(',','')
    return float(temp)

def process_amen(df_input):
    df = df_input.copy()
    if "amenities" in df.columns:
        #Keeping the length of amenities feature
        s = df.loc[:,"amenities"].apply(lambda x: x.split(', ')).apply(lambda x: len(x))
        df.loc[:,"amenities"] = s
    else:
        print("No feature 'amenities' was found. Returning unaffected DataFrame.")
    return df

def preprocess_dataframe(df_input):
    #Create local copy
    df = df_input.copy()

    #Drop non-categirical features and features with corr < 0.03.
    drop = ["id","host_id","listing_url","scrape_id","last_scraped","name","description",'neighborhood_overview',"picture_url","host_url",
            "host_name","host_location","host_about","host_thumbnail_url","host_picture_url","host_verifications",
            "neighbourhood","neighbourhood_group_cleansed","bathrooms", 'minimum_minimum_nights','maximum_minimum_nights', 'minimum_maximum_nights',
            'maximum_maximum_nights', 'minimum_nights_avg_ntm','maximum_nights_avg_ntm', "calendar_updated",
            "calendar_last_scraped","first_review","license","calculated_host_listings_count","bedrooms","beds","last_review","calculated_host_listings_count",
            "calculated_host_listings_count_entire_homes","calculated_host_listings_count_private_rooms","calculated_host_listings_count_shared_rooms",
            "host_neighbourhood","host_total_listings_count","property_type","host_acceptance_rate"]
    df.drop(labels = drop, axis = 1, inplace = True)
    
    #Keeping the length of "amenities" feature
    df = process_amen(df)

    #Turning "bathroom_texts" into 2 more useful features.
    df = engineer_bathroom_text(df)

    if "price" in df.columns:
        #Cleaning and renaming "price" feature to "target". Also moving it in the last place.
        df.insert(len(df.columns), "target", df.loc[:,"price"].apply(process_price))
        df.drop("price", axis = 1, inplace=True)

    #keeping only the year info from the "host_since" feature.
    df.loc[:,"host_since"] = pd.to_datetime(df.loc[:,"host_since"])
    df.loc[:,"host_since"] = df.loc[:,"host_since"].apply(lambda x: x.year)

    # Filling NaN values of "reviews_per_month" with 0, assuming there are none.
    df.loc[:,"reviews_per_month"].fillna(value = 0, inplace = True)

    #Filling NaN values of "host_response_rate" with 0% assuming there is no response rate.
    #Also we keep only the numeric part of the data as a float.
    df.loc[:,"host_response_rate"].fillna(value='0%',inplace=True)
    df.loc[:,"host_response_rate"] = df.loc[:,"host_response_rate"].apply(lambda x: float(x.split('%')[0]))


    ## Encoding categorical features

    #Encoding "room_type" feature in a scaling order.
    map_dict = {"Entire home/apt":3, "Private room":2,"Hotel room":1,"Shared room":0}
    df.loc[:,"room_type"] = df.loc[:,"room_type"].map(map_dict)

    #Encoding "host_response_time" feature in a scaling order. Also filling NaN values with 0 assuming there is no response and thus no response time.
    df.loc[:,"host_response_time"].fillna(value="0",inplace=True)
    map_dict = {"within an hour":4,"within a few hours":3,"within a day":2,"a few days or more":1, "0":0}
    df.loc[:,"host_response_time"] = df.loc[:,"host_response_time"].map(map_dict)

    #Encoding Bolean features.
    map_dict = {'t':1,'f':0}
    df.loc[:,"instant_bookable"] = df.loc[:,"instant_bookable"].map(map_dict)
    df.loc[:,"has_availability"] = df.loc[:,"has_availability"].map(map_dict)
    df.loc[:,"host_is_superhost"] = df.loc[:,"host_is_superhost"].map(map_dict)
    df.loc[:,"host_has_profile_pic"] = df.loc[:,"host_has_profile_pic"].map(map_dict)
    df.loc[:,"host_identity_verified"] = df.loc[:,"host_identity_verified"].map(map_dict)

    #Encoding "neighbourhood_cleansed" using the frequency of each value.
    map_dict = df.loc[:,"neighbourhood_cleansed"].value_counts().to_dict()
    df.loc[:,"neighbourhood_cleansed"] = df.loc[:,"neighbourhood_cleansed"].map(map_dict)

    #Checking if there are any NaN values remaining. If yes, return True.    
    return df, (df.isna().sum().sum() > 0)