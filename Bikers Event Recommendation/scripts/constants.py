class const(object):
    def __init__(self, data_dir):
        self.base_dir = data_dir

        # Data
        self.tours_df = None
        self.bikers_network_df = None
        self.tour_convoy_df = None
        self.train_df = None
        self.test_df = None
        self.bikers_df = None
        self.member_threshold = 5
        self.gps_threshold = 15
        self.location_recursion = 1
        self.current_step = 1
        self.total_steps = self.location_recursion + 3
        self.noise = []

        # Derived Data
        self.convoy_df = None
        self.tdf = None

        # Feature Extraction
        self.X = None
        self.number_of_topics = 25
        self.split_point = 12998
        self.cols = ['w' + str(i) for i in range(1, 101)] + ['w_other']
        self.learning_decay = 0.7
        self.learning_offset = 10
        self.batch_size = 128
        self.evaluate_every = -1
        self.random_state = 42
        self.max_iter = 10
        self.time_zone_feat = True
        self.preference_feat = False

        # Latent Dirichlet Allocation
        self.train_df_tr = None
        self.test_df_tr = None
        self.lda = None
        self.group = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.X_test = None
        self.train_group = None
        self.valid_group = None
        self.feat_cols = None

        # XGBoost
        self.xgbr = None
        self.pred = None
        self.pred_df = None
        self.best_params = {'objective': 'rank:map',
                            'learning_rate': 0.05,
                            'gamma': 1.0,
                            'booster': 'gbtree',
                            'max_depth': 5,
                            'n_estimators': 3250,
                            'reg_lambda': 1.0}
        self.submission_number = 1

        # Database
        self.locale_ = {
            'af_ZA': ['Afrikaans', 'South Africa', 'Cape Town', (-33.928992, 18.417396)],
            'gn_PY': ['Guarani', 'Paraguay', 'Asuncion', (-25.2800459, -57.6343814)],
            'ay_BO': ['Aymara', 'Bolivia', 'La Paz', (-16.4955455, -68.1336229)],
            'az_AZ': ['Azeri', 'Azerbaijan', 'Baku', (40.3754434, 49.8326748)],
            'id_ID': ['Indonesian', 'Indonesia', 'Jakarta', (-6.1753942, 106.827183)],
            'ms_MY': ['Malay', 'Malaysia', 'Kuala Lumpur', (3.1516964, 101.6942371)],
            'jv_ID': ['Javanese', 'Indonesia', 'Jakarta', (-6.1753942, 106.827183)],
            'bs_BA': ['Bosnian', 'Bosnia and Herzegovina', 'Sarajevo', (43.8519774, 18.3866868)],
            'ca_ES': ['Catalan', 'Spain', 'Madrid', (40.4167047, -3.7035825)],
            'cs_CZ': ['Czech', 'Czechia', 'Prague', (50.0874654, 14.4212535)],
            'ck_US': ['Cherokee', 'United States', 'Washington', (38.8950368, -77.0365427)],
            'cy_GB': ['Welsh', 'United Kingdom', 'London', (51.5073219, -0.1276474)],
            'da_DK': ['Danish', 'Denmark', 'Copenhagen', (55.6867243, 12.5700724)],
            'se_NO': ['Sami', 'Norway', 'Oslo', (59.9133301, 10.7389701)],
            'de_DE': ['German', 'Germany', 'Berlin', (52.5170365, 13.3888599)],
            'et_EE': ['Estonian', 'Estonia', 'Tallinn', (59.4372155, 24.7453688)],
            'en_IN': ['English (India)', 'India', 'New Delhi', (28.6138954, 77.2090057)],
            'en_PI': ['English (Pirate)', '', ''],
            'en_GB': ['English (UK)', 'United Kingdom', 'London', (51.5073219, -0.1276474)],
            'en_UD': ['English (Upside Down)', '', ''],
            'en_US': ['English (US)', 'United States', 'Washington', (38.8950368, -77.0365427)],
            'es_LA': ['Spanish', 'Laos', 'Vientiane', (17.9640988, 102.6133707)],
            'es_CL': ['Spanish (Chile)', 'Chile', 'Santiago', (-33.4377756, -70.6504502)],
            'es_CO': ['Spanish (Colombia)', 'Colombia', 'Bogota', (4.6533326, -74.083652)],
            'es_ES': ['Spanish (Spain)', 'Spain', 'Madrid', (40.4167047, -3.7035825)],
            'es_MX': ['Spanish (Mexico)', 'Mexico', 'Mexico City', (19.4326296, -99.1331785)],
            'es_VE': ['Spanish (Venezuela)', 'Venezuela', 'Caracas', (10.506098, -66.9146017)],
            'eo_EO': ['Esperanto', '', ''],
            'eu_ES': ['Basque', 'Spain', 'Madrid', (40.4167047, -3.7035825)],
            'tl_PH': ['Filipino', 'Philippines', 'Manila', (14.5907332, 120.9809674)],
            'fo_FO': ['Faroese', 'Faroe Islands', 'Torshavn', (62.012, -6.768)],
            'fr_FR': ['French (France)', 'France', 'Paris', (48.8566969, 2.3514616)],
            'fr_CA': ['French (Canada)', 'Canada', 'Toronto', (43.6534817, -79.3839347)],
            'fy_NL': ['Frisian', 'Netherlands', 'Amsterdam', (52.3727598, 4.8936041)],
            'ga_IE': ['Irish', 'Ireland', 'Dublin', (53.3497645, -6.2602732)],
            'gl_ES': ['Galician', 'Spain', 'Madrid', (40.4167047, -3.7035825)],
            'ko_KR': ['Korean', 'South Korea', 'Seoul', (37.5666791, 126.9782914)],
            'hr_HR': ['Croatian', 'Croatia', 'Zagreb', (45.8131847, 15.9771774)],
            'xh_ZA': ['Xhosa', 'South Africa', 'Cape Town', (-33.928992, 18.417396)],
            'zu_ZA': ['Zulu', 'South Africa', 'Cape Town', (-33.928992, 18.417396)],
            'is_IS': ['Icelandic', 'Iceland', 'Reykjavik', (64.145981, -21.9422367)],
            'it_IT': ['Italian', 'Italy', 'Rome', (41.8933203, 12.4829321)],
            'ka_GE': ['Georgian', 'Georgia', 'Tbilisi', (41.6934591, 44.8014495)],
            'sw_KE': ['Swahili', 'Kenya', 'Nairobi', (-1.3031689499999999, 36.826061224105075)],
            'tl_ST': ['Klingon', 'Sao Tome and Principe', 'Sao Tome', (0.3389242, 6.7313031)],
            'ku_TR': ['Kurdish', 'Turkey', 'Istanbul', (41.0096334, 28.9651646)],
            'lv_LV': ['Latvian', 'Latvia', 'Riga', (56.9493977, 24.1051846)],
            'fb_LT': ['Leet Speak', 'Lithuania', 'Vilnius', (54.6870458, 25.2829111)],
            'lt_LT': ['Lithuanian', 'Lithuania', 'Vilnius', (54.6870458, 25.2829111)],
            'li_NL': ['Limburgish', 'Netherlands', 'Amsterdam', (52.3727598, 4.8936041)],
            'la_VA': ['Latin', '', 'Vatican City', (41.9029, 12.451206)],
            'hu_HU': ['Hungarian', 'Hungary', 'Budapest', (47.48138955, 19.14607278448202)],
            'mg_MG': ['Malagasy', 'Madagascar', 'Antananarivo', (-18.9100122, 47.5255809)],
            'mt_MT': ['Maltese', 'Malta', 'Valletta', (35.8989818, 14.5136759)],
            'nl_NL': ['Dutch', 'Netherlands', 'Amsterdam', (52.3727598, 4.8936041)],
            'nl_BE': ['Dutch', 'Belgium', 'Brussels', (50.8465573, 4.351697)],
            'ja_JP': ['Japanese', 'Japan', 'Tokyo', (55.7746222, 37.6326806)],
            'nb_NO': ['Norwegian', 'Norway', 'Oslo', (59.9133301, 10.7389701)],
            'nn_NO': ['Norwegian', 'Norway', 'Oslo', (59.9133301, 10.7389701)],
            'uz_UZ': ['Uzbek', 'Uzbekistan', 'Tashkent', (41.3123363, 69.2787079)],
            'pl_PL': ['Polish', 'Poland', 'Warsaw', (52.2319581, 21.0067249)],
            'pt_BR': ['Portuguese (Brazil)', 'Brazil', 'Brasilia', (-10.3333333, -53.2)],
            'pt_PT': ['Portuguese (Portugal)', 'Portugal', 'Lisbon', (38.7077507, -9.1365919)],
            'qu_PE': ['Quechua', 'Peru', 'Lima', (-12.0621065, -77.0365256)],
            'ro_RO': ['Romanian', 'Romania', 'Bucharest', (44.4361414, 26.1027202)],
            'rm_CH': ['Romansh', 'Switzerland', 'Bern', (46.9482713, 7.4514512)],
            'ru_RU': ['Russian', 'Russian Federation', 'Moscow', (55.7504461, 37.6174943)],
            'sq_AL': ['Albanian', 'Albania', 'Tirana', (41.3305141, 19.825562857582966)],
            'sk_SK': ['Slovak', 'Slovakia', 'Bratislava', (48.1516988, 17.1093063)],
            'sl_SI': ['Slovenian', 'Slovenia', 'Ljubljana', (46.0499803, 14.5068602)],
            'so_SO': ['Somali', 'Somalia', 'Mogadishu', (2.0349312, 45.3419183)],
            'fi_FI': ['Finnish', 'Finland', 'Helsinki', (60.1674881, 24.9427473)],
            'sv_SE': ['Swedish', 'Sweden', 'Stockholm', (59.3251172, 18.0710935)],
            'th_TH': ['Thai', 'Thailand', 'Bangkok', (13.7544238, 100.4930399)],
            'vi_VN': ['Vietnamese', 'Viet Nam', 'Hanoi', (21.0294498, 105.8544441)],
            'tr_TR': ['Turkish', 'Turkey', 'Istanbul', (41.0096334, 28.9651646)],
            'zh_CN': ['Simplified Chinese (China)', 'China', 'Beijing', (39.9065084, 116.3912391)],
            'zh_TW': ['Traditional Chinese (Taiwan)', 'Taiwan', 'Taipei', (25.0375198, 121.5636796)],
            'zh_HK': ['Traditional Chinese (Hong Kong)', '', 'Hong Kong'],
            'el_GR': ['Greek', 'Greece', 'Athens', (37.9839412, 23.7283052)],
            'gx_GR': ['Classical Greek', 'Greece', 'Athens', (37.9839412, 23.7283052)],
            'be_BY': ['Belarusian', 'Belarus', 'Minsk', (53.902334, 27.5618791)],
            'bg_BG': ['Bulgarian', 'Bulgaria', 'Sofia', (42.6978634, 23.3221789)],
            'kk_KZ': ['Kazakh', 'Kazakhstan', 'Nur-Sultan', (51.1282205, 71.4306682)],
            'mk_MK': ['Macedonian', 'North Macedonia', 'Skopje', (41.9960924, 21.4316495)],
            'mn_MN': ['Mongolian', 'Mongolia', 'Ulaanbaatar', (47.9184676, 106.9177016)],
            'sr_RS': ['Serbian', 'Serbia', 'Belgrade', (44.8178131, 20.4568974)],
            'tt_RU': ['Tatar', 'Russian Federation', 'Moscow', (55.7504461, 37.6174943)],
            'tg_TJ': ['Tajik', 'Tajikistan', 'Dushanbe', (38.5762709, 68.7863573)],
            'uk_UA': ['Ukrainian', 'Ukraine', 'Kiew', (50.4500336, 30.5241361)],
            'hy_AM': ['Armenian', 'Armenia', 'Yerevan', (40.1776121, 44.5125849)],
            'yi_DE': ['Yiddish', 'Germany', 'Berlin', (52.5170365, 13.3888599)],
            'he_IL': ['Hebrew', 'Israel', 'Tel Aviv', (32.0852997, 34.7818064)],
            'ur_PK': ['Urdu', 'Pakistan', 'Islamabad', (33.6938118, 73.0651511)],
            'ar_AR': ['Arabic', 'Argentina', 'Buenos Aires', (-34.6117879, -58.494934)],
            'ps_AF': ['Pashto', 'Afghanistan', 'Kabul', (34.5406819, 69.0425447)],
            'fa_IR': ['Persian', 'Iran', 'Tehran', (35.6892523, 51.3896004)],
            'sy_SY': ['Syriac', 'Syria', 'Damascus', (33.5130695, 36.3095814)],
            'ne_NP': ['Nepali', 'Nepal', 'Kathmandu', (27.708317, 85.3205817)],
            'mr_IN': ['Marathi', 'India', 'Mumbai', (19.0759899, 72.8773928)],
            'sa_IN': ['Sanskrit', 'India', 'New Delhi', (28.6138954, 77.2090057)],
            'hi_IN': ['Hindi', 'India', 'New Delhi', (28.6138954, 77.2090057)],
            'bn_IN': ['Bengali', 'India', 'Kolkata', (22.5414185, 88.35769124388872)],
            'pa_IN': ['Punjabi', 'India', 'Amritsar', (31.6343083, 74.8736788)],
            'gu_IN': ['Gujarati', 'India', 'Ahmedabad', (23.0216238, 72.5797068)],
            'ta_IN': ['Tamil', 'India', 'Chennai', (13.0836939, 80.270186)],
            'te_IN': ['Telugu', 'India', 'Hyderabad', (17.38878595, 78.46106473453146)],
            'kn_IN': ['Kannada', 'India', 'Banglore', (16.02595815, 80.9161541112084)],
            'ml_IN': ['Malayalam', 'India', 'Thiruvananthapuram', (8.576970549999999, 77.05012463730725)],
            'km_KH': ['Khmer', 'Cambodia', 'Phnom Penh', (11.568271, 104.9224426)]
        }
