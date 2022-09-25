## greet + show_phones
* greet   
  - utter_how_can_I_help
* buy_phone_laptop{"category":"phone"}
  - utter_ask_ram
* give_information{"ram":"4 GB"}
  - utter_ask_camera
* give_information{"camera":"40 megapixel"}
  - utter_ask_battery
* give_information{"battery":"3000 mah"}
  - action_search
* goodbye
  - utter_goodbye

## greet + show_latest_news
* greet
  - utter_how_can_I_help
* latest_news_phones_laptops{"category":"phone"}
  - action_show_latest_news
* goodbye
  - utter_goodbye

