API_KEY = ''
GEMINI_KEY = ''
COHERE_KEY = ''

FORMAT_LONG = '''{
  "sightingDetails": {
    "DenseNarrativeSection": "Short Free-form dense narrative description of the event from the omniscient perspective, including any relevant information ",
    "trustScore" : "A reliability score from 0-100 based on the UAP/UFO report",
    "date": "YYYY-MM-DD",
    "timeOfDay": "HH:MM (24-hour format)",
    "location": {
      "name": "Specific Location",
      "latitude": "Decimal Degrees",
      "longitude": "Decimal Degrees"
    },
    "weatherConditions": {
      "generalDescription": "Description of weather",
      "visibility": "Clear, Partially Cloudy, etc.",
      "cloudCover": "Percentage or description",
      "extremePhenomena": "Any extreme weather conditions"
    },
    "sightingDuration": "Duration in hours or minutes",
    "uapCharacteristics": {
      "size": "Description or comparison",
      "shape": "The Shape of the object",
      "color": "Description of colors and patterns",
      "altitude": "Estimated or measured altitude",
      "speed": "Estimated or measured speed",
      "movementPattern": "Zigzagging, straight line, stationary, etc.",
      "changeInAppearance": "Details of any transformations",
      "sound": "Description of any associated sounds",
      "electromagneticEffects": "Any disturbances in electronic devices",
      "visualClarity": "Clarity of object's appearance",
      "detailedStructure": "Description of surface, texture, etc.",
      "presenceHumanoids": "Presence of humanoids (Yes/No)",
      "presenceSounds": "Presence of sound (Yes/No)",
      "presenceLights": "Presence of lights (Yes/No)"
    },
    "observerDetails": {
      "numberOfWitnesses": "Number of witnesses",
      "observerBackground": {
        "occupation": "Occupation",
        "experienceWithAircraft": "Experience with aircraft or aerial phenomena"
      },
      "psychologicalEffects": "Psychological/Emotional effects on observer",
      "emotionalStateBeforeSighting": "Observer's emotional state before sighting",
      "behavioralResponse": "Observer's reaction during the sighting",
      "subsequentEffects": "Physical or psychological effects post-sighting"
    },
    "evidence": {
      "photographic": "Available/Not available",
      "video": "Available/Not available",
      "url": "URL"
    },
    "additionalInformation": {
      "interactionWithEnvironment": "Effects on surroundings",
      "wildlifeBehavior": "Any unusual animal behavior",
      "previousActivity": "History of similar sightings in the area",
      "flightCharacteristics": "Detailed flight pattern description",
      "radarData": "Available/Not available",
      "airTrafficCommunications": "Reports from nearby aircraft or control towers",
      "officialResponses": "Governmental or scientific responses",
      "historicalComparisons": "Reference to similar historical sightings",
      "communicationAttempts": "Perceived attempts at communication from UAP",
      "temporalAnomalies": "Reports of lost time or time discrepancies",
      "anomalyDuration": "Specific times for changes in UAP appearance/behavior",
      "sightingFrequency": "Frequency of sightings by the observer"
    }
  }
}'''

# ---------------------------------------------------------------------------
# Spreadsheet-mapped format for UAP Activities Data 1975 Onwards.xlsx
# Used by uap_xlsx_filler.py — structure mirrors the spreadsheet column groups.
# ---------------------------------------------------------------------------

FACILITY_TYPES = [
    "Aircraft Testing",
    "Atomic weapons assembly facilities",
    "Atomic Weapons Deployment",
    "Atomic weapons stockpile sites",
    "Facility type unknown",
    "ICBM Minuteman III MRV",
    "ICBM Sites",
    "Materials Production",
    "Military Base",
    "Missile Testing",
    "Other facility type/function not listed below",
    "Population Centre",
    "Radioactive materials production plants",
    "Reactor Facility",
    "Space Flight",
    "Weapons Deployment Base",
    "Weapons Design & Assembly",
    "Weapons Stockpile",
]

FORMAT_LONG_XLSX = {
    # ── Source ─────────────────────────────────────────────────────────────────
    # Maps to spreadsheet columns B-D
    "source": {
        "name":      "string — name of the reporting database, publication or organisation (e.g. NICAP, MUFON, AFOSI)",
        "ref":       "string — specific citation, case ID or URL for this case",
        "duplicate": "string — leave blank unless this is a known duplicate; if so, note the original case reference"
    },

    # ── Date / Time ────────────────────────────────────────────────────────────
    # Maps to spreadsheet columns E-M
    "date_time": {
        "year":            "integer YYYY — calendar year of the sighting (e.g. 1975); null if unknown",
        "month":           "integer 1-12 — calendar month; null if unknown",
        "day":             "integer 1-31 — calendar day; null if unknown",
        "day_night":       "enum [D, N, U] — D = daytime, N = night-time, U = unknown",
        "local_time":      "string HH:MM — local 24-hour start time of the sighting (e.g. 20:30); empty if unknown",
        "local_time_code": "string — IANA timezone abbreviation or UTC offset (e.g. EST, UTC+2, PST)",
        "duration_min":    "integer — total observed duration in minutes; null if unknown",
        "gmt_start":       "string HH:MM — start time converted to GMT/UTC; empty if unknown",
        "description":     "string — free-text date/time context when exact values cannot be determined"
    },

    # ── Location ───────────────────────────────────────────────────────────────
    # Maps to spreadsheet columns N-T
    "location": {
        "country":     "string — ISO 3166-1 alpha-2 code or full name (e.g. US, UK, France)",
        "state":       "string — state, province or region abbreviation or name (e.g. TN, Ontario)",
        "city":        "string — city, town or nearest known populated place",
        "latitude":    "float decimal degrees WGS-84, positive = North (e.g. 36.010); null if unknown",
        "longitude":   "float decimal degrees WGS-84, positive = East (e.g. -84.267); null if unknown",
        "description": "string — additional location context: landmarks, base name, access restrictions, etc.",
        "type":        "enum [Atomic site, Military base, Population centre, Waste facility, Water body, River, Airport, Forest, Desert, Ocean, Unknown] — primary character of the location"
    },

    # ── Witness ────────────────────────────────────────────────────────────────
    # Maps to spreadsheet columns U-W
    "witness": {
        "count":       "integer — total number of independent witnesses; null if unknown",
        "type":        "string — comma-separated subset of [Military, Public, Police, Pilot, Civilian, Scientist] — witness categories",
        "description": "string — additional detail about witness background, rank, occupation or credibility assessment"
    },

    # ── Investigation ──────────────────────────────────────────────────────────
    # Maps to spreadsheet columns X-Y
    "investigation": {
        "source":      "string — organisation or individual that formally investigated (e.g. USAF Blue Book, NICAP, MUFON)",
        "description": "string — summary of investigation process and conclusions"
    },

    # ── Craft ──────────────────────────────────────────────────────────────────
    # Maps to spreadsheet columns Z-AD
    "craft": {
        "primary_shape":   "enum [Sphere, Disc, Cigar, Triangle, Diamond, Cylinder, Boomerang, Chevron, Egg, Cone, Saturn, Cross, Rectangle, Teardrop, Fireball, Light, Orb, Tic-Tac, Cube, Pyramid, Unknown, Other] — dominant observed shape",
        "secondary_shape": "enum [same values as primary_shape, or empty] — secondary or sub-shape if the object had a compound form",
        "colour":          "string — colour or colour combination observed (e.g. Silver, Glowing orange, Red and white flashing)",
        "size":            "enum [Tiny (<0.5 m), Small (0.5-3 m), Medium (3-10 m), Large (10-50 m), Very large (50-200 m), Massive (>200 m), Unknown] — estimated physical size at closest approach",
        "description":     "string — free-text shape, texture, surface markings, lighting and structural detail"
    },

    # ── Performance / 5 Observables ────────────────────────────────────────────
    # Maps to spreadsheet columns AE-AK
    # The 5 Observables are the Pentagon/AATIP framework for anomalous performance
    "performance": {
        "speed_mph":                  "float — estimated or reported speed in miles per hour; null if unknown",
        "acceleration_g":             "float — observed peak acceleration in g-force units; null if unknown",
        "hypersonic":                 "enum [Y, ] — Y if hypersonic velocity without acoustic signature or thermal bloom was observed; blank otherwise",
        "instantaneous_acceleration": "enum [Y, ] — Y if instantaneous or near-instantaneous acceleration (impossible for known craft) was observed; blank otherwise",
        "low_observability":          "enum [Y, ] — Y if cloaking, active stealth or anomalously low radar cross-section was noted; blank otherwise",
        "trans_medium_travel":        "enum [Y, ] — Y if seamless travel across air, water or near-space boundaries was observed; blank otherwise",
        "positive_lift":              "enum [Y, ] — Y if antigravitic or non-aerodynamic lift (no wings, rotors, visible propulsion) was observed; blank otherwise"
    },

    # ── Military Context ───────────────────────────────────────────────────────
    # Maps to spreadsheet columns AL-AP
    "military": {
        "reported_by":     "string — comma-separated subset of [Military, Public, Police, Pilot, Intelligence] — who officially reported or filed the case",
        "military_public": "enum [Military, Public, Mixed] — whether the primary location/context is military-controlled or publicly accessible",
        "facility_name":   "string — name of the specific military base, nuclear site or government facility involved; empty if not applicable",
        "facility_type":   "enum [Aircraft Testing, Atomic weapons assembly facilities, Atomic Weapons Deployment, Atomic weapons stockpile sites, Facility type unknown, ICBM Minuteman III MRV, ICBM Sites, Materials Production, Military Base, Missile Testing, Other facility type/function not listed below, Population Centre, Radioactive materials production plants, Reactor Facility, Space Flight, Weapons Deployment Base, Weapons Design & Assembly, Weapons Stockpile] — closest matching SCU facility category",
        "comments":        "string — source notes on the military relevance or sensitivity of the case"
    },

    # ── Effects ────────────────────────────────────────────────────────────────
    # Maps to spreadsheet columns AQ-AT
    "effects": {
        "atomic_related":   "enum [Y, N] — Y if the event involved an atomic/nuclear site, material or weapon",
        "communication":    "enum [Y, N] — Y if communication interference, anomalous signals or transmissions were reported",
        "physical_effects": "enum [Y, N] — Y if physical effects on people, animals, vehicles or the local environment were reported",
        "text":             "string — free-text description of all communication anomalies and physical effects observed"
    },

    # ── Engagement flags (spreadsheet group "Engagement", cols AU-BE) ──────────
    # These are independent Y/N observed-fact flags.
    # Multiple Y values are allowed simultaneously.
    "engagement_flags": {
        "aircraft_engagement":         "enum [Y, N] — Y if military or civilian aircraft actively manoeuvred to intercept or engage the UAP",
        "aircraft_encounters":         "enum [Y, N] — Y if aircraft encountered the UAP without deliberate engagement",
        "active_radar_jamming":        "enum [Y, N] — Y if active radar jamming — of the UAP or by the UAP — was documented",
        "over_military_installation":  "enum [Y, N] — Y if the UAP was observed directly over a military installation or restricted airspace",
        "during_missile_test":         "enum [Y, N] — Y if the sighting coincided with a missile, rocket or high-altitude balloon test",
        "radar_tracking":              "enum [Y, N] — Y if the UAP was tracked on military or civilian radar",
        "radio_interference":          "enum [Y, N] — Y if radio or audio receiver interference (noise, cutouts) was reported",
        "radar_jamming":               "enum [Y, N] — Y if radar receivers or displays were jammed or degraded",
        "directed_radar":              "enum [Y, N] — Y if the UAP emitted directed radar transmissions mimicking in-use frequencies",
        "coded_radar":                 "enum [Y, N] — Y if coded radar (IFF — Identification Friend or Foe) transmissions were detected from the UAP",
        "multiple_interactive_flight": "enum [Y, N] — Y if multiple UAP performed coordinated or interactive flight manoeuvres with each other"
    },

    # ── Engagement Type (spreadsheet group "Engagement Type", cols BF-BO) ──────
    # CONSTRAINT: exactly ONE field must be set to "P" (Primary — the dominant
    # engagement type for this case). Additional fields may be "S" (Secondary —
    # also present but not the defining characteristic). All remaining fields must
    # be left blank (empty string). Do NOT use Y/N here.
    #
    # Rule: 1 × P required | 0-many × S allowed | blank = not applicable
    # Exception: if NO engagement type applies at all, set no_engagement = "P"
    # and leave every other field blank.
    "engagement_type": {
        "interactive_flight":       "P, S, or blank — UAP performed flight that appeared to react to or interact with observers or aircraft",
        "radical_flight":           "P, S, or blank — UAP executed physically impossible manoeuvres (right-angle turns, instant acceleration, sudden reversal)",
        "loitering":                "P, S, or blank — UAP hovered, circled or loitered in a defined area for an extended period",
        "electronic_transmissions": "P, S, or blank — electronic signals or transmissions attributed to the UAP were detected",
        "interference_weapons":     "P, S, or blank — weapons systems (guns, missiles, guidance electronics) were interfered with or disabled",
        "military_intrusions":      "P, S, or blank — UAP physically intruded into a secured military installation or restricted airspace",
        "occupant_encounter":       "P, S, or blank — occupants of the UAP were directly encountered by witnesses (physical contact or attempted communication)",
        "occupant_observed":        "P, S, or blank — occupants were visually observed from a distance but no encounter occurred",
        "close_approach":           "P, S, or blank — UAP made a close approach (<30 m) to witnesses, vehicles or structures",
        "no_engagement":            "P, S, or blank — set to P only when no other engagement type applies; mutually exclusive with all other P/S assignments"
    },

    # ── Case Narrative ─────────────────────────────────────────────────────────
    # Maps to spreadsheet column BP
    "case_text": {
        "text": "string — verbatim or faithfully summarised case narrative from the original source"
    }
}

# ---------------------------------------------------------------------------
# FORMAT_SCU_V2 — "SCU v2" parsing schema
#
# A redlined evolution of FORMAT_LONG_XLSX. It adds the parser-emitted
# source / assessment / anomaly blocks, tri-state (Y/N/U) flags, structured
# witness roles and an evidence array — everything the SCU five-criterion
# eligibility gate needs at parse time.
#
# The `manifest` block is a DOWNSTREAM JOIN target keyed on source.documentId:
# the LLM must leave every manifest field null (each field is described as
# "null — populated by manifest join").
#
# Pair the parsed output with scu_normalizer.py to canonicalise the values and
# derive the scu_eligible gate columns.
# ---------------------------------------------------------------------------
FORMAT_SCU_V2 = {
    # ── Sighting details (kept from FORMAT_LONG, tightened) ────────────────────
    "sightingDetails": {
        "DenseNarrativeSection": "string — short omniscient free-form narrative of the event, including any relevant context",
        "trustScore": "integer 0-100 — composite reliability score for the DATA (not the phenomenon): source provenance, completeness of date/time/location/description, investigator class, timeliness, observation modality, witness count and independence",
        "trustScoreNotes": "string — one-sentence justification listing which dimensions drove trustScore",
        "date": "YYYY-MM-DD",
        "timeOfDay": "HH:MM (24-hour format)",
        "location": {
            "name": "string — specific location",
            "latitude": "float — decimal degrees; null if unknown",
            "longitude": "float — decimal degrees; null if unknown"
        },
        "weatherConditions": {
            "generalDescription": "string — description of weather",
            "visibility": "string — Clear, Partially Cloudy, etc.",
            "cloudCover": "string — percentage or description",
            "extremePhenomena": "string — any extreme weather conditions"
        },
        "sightingDuration": "string — duration in hours or minutes",
        "uapCharacteristics": {
            "size": "string — description or comparison",
            "shape": "string — shape of the object",
            "color": "string — colours and patterns",
            "altitude": "string — estimated or measured altitude",
            "speed": "string — estimated or measured speed",
            "movementPattern": "string — zig-zag, straight line, stationary, etc.",
            "changeInAppearance": "string — details of any transformations",
            "sound": "string — description of any associated sounds",
            "electromagneticEffects": "string — disturbances in electronic devices",
            "visualClarity": "string — clarity of the object's appearance",
            "detailedStructure": "string — surface, texture, structural detail",
            "presenceHumanoids": "enum [Y, N, U] — Y ONLY if humanoid beings are described in, on, emerging from, or immediately beside the object itself; ordinary humans (witnesses, bystanders, aircraft pilots, a drone's operator on the ground) never make this Y. N when the source explicitly notes none; U when not mentioned",
            "presenceHumanoids_notes": "string — qualifiers (e.g. 'reported via telepathy', 'in photograph'); empty if none",
            "presenceSounds": "enum [Y, N, U]",
            "presenceLights": "enum [Y, N, U]"
        },
        "observerDetails": {
            "numberOfWitnesses": "integer — exact witness count when reported; null otherwise",
            "numberOfWitnessesFreeform": "string — original phrasing when non-numeric (e.g. 'Multiple', 'Hundreds', 'At least 6')",
            "observerBackground": {
                "occupation": "string — observer occupation",
                "experienceWithAircraft": "string — experience with aircraft or aerial phenomena"
            },
            "psychologicalEffects": "string — psychological/emotional effects on the observer",
            "emotionalStateBeforeSighting": "string — observer's emotional state before the sighting",
            "behavioralResponse": "string — observer's reaction during the sighting",
            "subsequentEffects": "string — physical or psychological effects after the sighting"
        },
        "evidence": [
            {
                "mediaType": "enum [photographic, video, radar_trace, audio, instrumented, physical_sample, sketch]",
                "availability": "enum [available, not_available, attempted_failed, referenced_not_seen, inconclusive]",
                "url": "string — direct link if available, else empty",
                "description": "string — qualifiers (e.g. '8mm film exposed 10 seconds', 'sketch by witness')",
                "chainOfCustody": "string — provenance notes if recorded"
            }
        ],
        "additionalInformation": {
            "interactionWithEnvironment": "string — effects on surroundings",
            "wildlifeBehavior": "string — unusual animal behaviour",
            "previousActivity": "string — history of similar sightings in the area",
            "flightCharacteristics": "string — detailed flight-pattern description",
            "radarData": "enum [available, not_available, available_with_caveat, unknown]",
            "radarDataNotes": "string — qualifiers (e.g. 'tracked by theodolite', 'controller had no contact')",
            "airTrafficCommunications": "string — reports from nearby aircraft or control towers",
            "officialResponses": "string — governmental or scientific responses",
            "historicalComparisons": "string — reference to similar historical sightings",
            "communicationAttempts": "string — perceived attempts at communication from the UAP",
            "temporalAnomalies": "string — reports of lost time or time discrepancies",
            "anomalyDuration": "string — specific times for changes in UAP appearance/behaviour",
            "sightingFrequency": "string — frequency of sightings by the observer"
        }
    },

    # ── Source (parser-emitted; expanded provenance + adjudication) ────────────
    "source": {
        "name": "string — reporting database, publication or organisation (e.g. NICAP, MUFON, AFOSI)",
        "ref": "string — specific citation, case ID or URL for this case",
        "duplicate": "string — case-ref of the original if this is a known duplicate, else blank",
        "documentId": "string — identifier of the upstream source document; copy if visible, else blank",
        "pages": "string — page range within the source document (e.g. 'page_0004-page_0006'); blank if unknown",
        "chunkCount": "integer — number of text chunks contributing to this row; null if unknown",
        "parseWarning": "boolean — true if a parse error was flagged on this row; default false",
        "rawText": "string — un-truncated narrative text of the source",
        "agency": "string — coarse agency taxonomy inferred from the document (FBI, NARA-CIA, DOD, MISC, NASA); null if not indicated",
        "collection": "string — sub-collection within the agency (e.g. 'series-38', 'hs1-834228961'); null if not indicated",
        "tier": "enum [primary_investigative, secondary_investigative, reference_book, press, retrospective_personal_account, unknown] — coarse reliability tier; default 'unknown'",
        "firstRecordedDate": "YYYY-MM-DD — earliest dateline visible in the document text (cable/letter/memo date); null if not visible",
        "contemporaneous": "enum [Y, N, U] — Y if firstRecordedDate is within one year of the sighting, N if later, U if either date is unknown",
        "originalAdjudication": "string — verbatim conclusion by the investigating authority as quoted in the document; null if no adjudication appears",
        "originalAdjudicationType": "enum [unidentified, explained, dismissed, no_investigation, not_stated] — coded reading of the adjudication; use 'not_stated' when the document is silent"
    },

    # ── Manifest (DOWNSTREAM JOIN ONLY — leave every field null) ───────────────
    "manifest": {
        "pdfUrl": "null — populated by manifest join, never by the parser",
        "releaseDate": "null — populated by manifest join, never by the parser",
        "isRedacted": "null — populated by manifest join, never by the parser",
        "documentType": "null — populated by manifest join, never by the parser",
        "documentDescription": "null — populated by manifest join, never by the parser",
        "agencyRelease": "null — populated by manifest join, never by the parser",
        "videoId": "null — populated by manifest join, never by the parser",
        "modalImage": "null — populated by manifest join, never by the parser",
        "joinStatus": "null — populated by manifest join, never by the parser"
    },

    # ── Assessment (parse-time judgement about the report's coherence) ─────────
    "assessment": {
        "contradictsUap": "boolean — true if the source itself states the incident was NOT a UAP (film defect, balloon, weather, hoax, sarcastic/diplomatic usage)",
        "notes": "string — short justification for contradictsUap, or any other parse-time annotation"
    },

    # ── Anomaly (SCU Criterion 3 — anomalous characterization) ─────────────────
    "anomaly": {
        "structure": "boolean — anomalous shape, size, surface or detailed structure",
        "flight": "boolean — anomalous flight (hypersonic without sonic boom, instantaneous acceleration, trans-medium, etc.)",
        "occupant": "boolean — occupants observed or encountered",
        "signal": "boolean — anomalous electronic transmissions or radio behaviour",
        "summary": "string — one-sentence justification for any of the above being true"
    },

    # ── Date / Time ────────────────────────────────────────────────────────────
    "date_time": {
        "year": "integer YYYY — calendar year; null if unknown",
        "month": "integer 1-12 — calendar month; null if unknown",
        "day": "integer 1-31 — calendar day; null if unknown",
        "day_night": "enum [D, N, U]",
        "local_time": "string HH:MM — local 24-hour start time; empty if unknown",
        "local_time_code": "string — IANA timezone abbreviation or UTC offset (e.g. EST, UTC+2)",
        "duration_min": "integer — total observed duration in minutes; null if unknown",
        "gmt_start": "string HH:MM — start time converted to GMT/UTC; empty if unknown",
        "description": "string — free-text date/time context when exact values cannot be determined",
        "yearUncertainty": "enum [exact, approximate_year, approximate_decade, unknown]",
        "dateUncertainty": "enum [exact, approximate_month, approximate_year, unknown]",
        "inWindow_1945_1975": "boolean — true if the sighting falls within the SCU master study window"
    },

    # ── Location ───────────────────────────────────────────────────────────────
    "location": {
        "country": "string — ISO 3166-1 alpha-2 code (strict). For non-country locations use sentinels: INTL_WATERS, MOON, AQ, MULTIPLE, SCAND, UNKNOWN",
        "state": "string — US Census two-letter code for US states; ISO 3166-2 subdivision code elsewhere; empty if not applicable",
        "city": "string — city, town or nearest known populated place",
        "latitude": "float — decimal degrees WGS-84, positive = North; null if unknown",
        "longitude": "float — decimal degrees WGS-84, positive = East; null if unknown",
        "description": "string — additional location context: landmarks, base name, access restrictions",
        "type": "enum [Atomic site, Military base, Population centre, Waste facility, Water body, River, Airport, Forest, Desert, Ocean, Space flight, Mixed, Other, Rural, Missile testing, Unknown] — primary character of the location",
        "secondaryType": "enum [same values as type, or empty] — for locations that genuinely span two types"
    },

    # ── Witness (structured roles array) ───────────────────────────────────────
    "witness": {
        "count": "integer — exact number of independent witnesses; null if unknown",
        "countFreeform": "string — original phrasing when non-numeric (e.g. 'Multiple', 'Hundreds')",
        "roles": ["enum [Military, Public, Police, Pilot, Civilian, Scientist, Intelligence, Astronaut, Politician, Security] — JSON array of witness roles; no duplicates, order irrelevant. Roles describe the human WITNESSES themselves ('Pilot' = a pilot witnessed the event), never the object's or a drone's operator"],
        "notes": "string — rank/role qualifiers that don't belong in the roles enum (e.g. 'Naval Reserve')",
        "description": "string — broader witness background, credibility assessment, observer-location detail"
    },

    # ── Investigation (SCU Criterion 1 + 4) ────────────────────────────────────
    "investigation": {
        "source": "string — organisation or individual that investigated",
        "investigatorClass": "enum [USAF, NICAP, MUFON, FBI, CIA, NASA, AARO, law_enforcement, civilian_research, press_only, none, other]",
        "startDate": "YYYY-MM-DD — when the investigation began; null if unknown",
        "lagDays": "integer — days between sighting and start of investigation; null if unknown",
        "timeliness": "enum [timely, late, unknown] — timely = lagDays <= 365 (SCU Criterion 1); late = lagDays > 365",
        "reports_within_1_month_of_sighting": "enum [Y, N, U] — Y if the report or field investigation was initiated within one month of the observation; finer-grained than the SCU Criterion 1 one-year rule",
        "reports_within_1_year_of_sighting": "enum [Y, N, U] — Y if the report or field investigation was initiated within one year of the observation (SCU Criterion 1, the one-year timeliness rule)",
        "description": "string — summary of the investigation process and conclusions"
    },

    # ── Craft ──────────────────────────────────────────────────────────────────
    "craft": {
        "primary_shape": "enum [Sphere, Disc, Cigar, Triangle, Diamond, Cylinder, Boomerang, Chevron, Egg, Cone, Saturn, Cross, Rectangle, Teardrop, Fireball, Light, Orb, Tic-Tac, Cube, Pyramid, Oval, Dome, Crescent, Unknown, Other] — dominant observed shape; fold Round/Circle/Ball/Spherical into Sphere. Purely descriptive of the reported appearance, independent of identification (a balloon seen as a sphere is Sphere; the prosaic explanation belongs in assessment.contradictsUap). If the source names a conventional craft (airplane, helicopter, quadcopter drone) without describing one of the listed geometries, use Other — never map a named conventional craft to Disc/Cigar/etc. by analogy",
        "secondary_shape": "enum [same values as primary_shape, or empty]",
        "colour": "string — colour or colour combination observed",
        "size": "enum [Tiny (<0.5 m), Small (0.5-3 m), Medium (3-10 m), Large (10-50 m), Very large (50-200 m), Massive (>200 m), Unknown]",
        "sizeRangeMeters": "string — explicit metre range derived from the size band (e.g. '3-10 m'); empty if Unknown",
        "description": "string — free-text shape, texture, surface markings, lighting and structural detail"
    },

    # ── Performance / 5 Observables (tri-state) ────────────────────────────────
    "performance": {
        "speed_mph": "float — estimated or reported speed in mph; null if unknown",
        "acceleration_g": "float — observed peak acceleration in g; null if unknown",
        "hypersonic": "enum [Y, N, U] — Y ONLY if speed above ~Mach 5 is reported or clearly implied (typically without sonic boom); N if observed speeds were conventional; U if not assessed",
        "instantaneous_acceleration": "enum [Y, N, U] — Y ONLY if the object accelerated from rest or slow motion to extreme speed near-instantaneously, beyond any known aircraft; a bird, insect or drone darting at close range is N; U if not assessed",
        "low_observability": "enum [Y, N, U] — Y ONLY if the object evaded detection when detection was expected (no radar return despite visual contact, abrupt vanishing, cloaking-like behaviour); an object merely too small or too distant for radar is N; U if not assessed",
        "trans_medium_travel": "enum [Y, N, U] — Y ONLY if the object crossed between media (air/water/space) without apparent performance loss; a diving bird or a missile splashing down is N; U if no crossing was observed",
        "positive_lift": "enum [Y, N, U] — Y ONLY if the object holds or gains altitude with NO apparent means of lift or propulsion (no wings, rotors, jets, or balloon envelope). Any aircraft, drone or balloon flying normally is N — merely being airborne or hovering is NOT positive lift. U when the lift mechanism cannot be judged from the source"
    },

    # ── Military context ───────────────────────────────────────────────────────
    "military": {
        "reported_by": ["enum [Military, Public, Police, Pilot, Intelligence] — JSON array of who officially reported the case"],
        "military_public": "enum [Military, Public, Mixed, None] — use None when the military/public character of the location is unknown",
        "facility_name": "string — name of the military base, nuclear site or government facility; empty if not applicable",
        "facility_type": "enum [Aircraft testing, Atomic weapons assembly facilities, Atomic weapons deployment, Atomic weapons stockpile sites, Facility type unknown, ICBM Minuteman III MRV, ICBM Sites, Materials Production, Military base, Missile testing, Other, Population centre, Radioactive materials production plants, Reactor Facility, Space flight, Weapons Deployment Base, Weapons Design & Assembly, Weapons Stockpile, Airport, Unknown, None] — use None when the facility type is unknown",
        "comments": "string — source notes on the military relevance or sensitivity of the case"
    },

    # ── Effects (tri-state) ────────────────────────────────────────────────────
    "effects": {
        "atomic_related": "enum [Y, N, U] — Y if the event involved an atomic/nuclear site, material or weapon",
        "communication": "enum [Y, N, U] — Y if communication interference or anomalous signals were reported",
        "physical_effects": "enum [Y, N, U] — Y if physical effects on people, animals, vehicles or environment were reported",
        "text": "string — free-text description of communication anomalies and physical effects"
    },

    # ── Engagement flags (tri-state observed-fact flags) ───────────────────────
    "engagement_flags": {
        "aircraft_engagement": "enum [Y, N, U]",
        "aircraft_encounters": "enum [Y, N, U]",
        "active_radar_jamming": "enum [Y, N, U]",
        "over_military_installation": "enum [Y, N, U]",
        "during_missile_test": "enum [Y, N, U]",
        "radar_tracking": "enum [Y, N, U]",
        "radio_interference": "enum [Y, N, U]",
        "radar_jamming": "enum [Y, N, U]",
        "directed_radar": "enum [Y, N, U]",
        "coded_radar": "enum [Y, N, U]",
        "multiple_interactive_flight": "enum [Y, N, U]"
    },

    # ── Engagement type (exactly one P; 0-many S; blank otherwise) ─────────────
    "engagement_type": {
        "interactive_flight": "P, S, or blank — UAP flight appeared to react to or interact with observers or aircraft",
        "radical_flight": "P, S, or blank — physically impossible manoeuvres (right-angle turns, instant acceleration, sudden reversal)",
        "loitering": "P, S, or blank — UAP hovered, circled or loitered for an extended period",
        "electronic_transmissions": "P, S, or blank — electronic signals attributed to the UAP were detected",
        "interference_weapons": "P, S, or blank — weapons systems were interfered with or disabled",
        "military_intrusions": "P, S, or blank — UAP intruded into a secured installation or restricted airspace",
        "occupant_encounter": "P, S, or blank — occupants OF THE OBJECT were directly encountered by witnesses. 'Occupant' means a being belonging to the object; a drone operator, conventional-aircraft pilot or ground crew is NOT an occupant — leave blank for those",
        "occupant_observed": "P, S, or blank — occupants OF THE OBJECT were visually observed from a distance, no encounter. Same rule as occupant_encounter: drone operators, aircraft pilots and ground personnel are NOT occupants — leave blank for those",
        "close_approach": "P, S, or blank — UAP made a close approach (<30 m) to witnesses, vehicles or structures",
        "no_engagement": "P, S, or blank — set to P only when no other engagement type applies"
    },

    # ── Case narrative ─────────────────────────────────────────────────────────
    "case_text": {
        "text": "string — faithfully summarised case narrative; the verbatim full text lives in source.rawText"
    }
}

# ---------------------------------------------------------------------------
# FORMAT_SCU_V3 — "SCU v3" parsing schema
#
# A redlined evolution of FORMAT_SCU_V2 that folds in the UFOSETI assessment
# frameworks (UFOCAT-style explanation taxonomy + Hynek-Vallée AN/CE recovery
# and biological-entity flags + medical-effect human-injury extension).
#
# Adds, relative to FORMAT_SCU_V2:
#   • assessment.explanationCategory — UFOCAT-style conventional-explanation taxonomy
#   • anomaly.validated / materialRecovery / biologicalEntity — official-body review,
#     craft-debris recovery, and non-human entity encounter flags
#   • effects.humanInjury / humanInjuryNotes — explicit medical-injury extension
#
# The `manifest` block remains a DOWNSTREAM JOIN target keyed on source.documentId:
# the LLM must leave every manifest field null.
#
# Pair the parsed output with scu_normalizer.py to canonicalise the values and
# derive the scu_eligible gate columns.
# ---------------------------------------------------------------------------
FORMAT_SCU_V3 = {
    # ── Sighting details (kept from SCU_v2, unchanged) ─────────────────────────
    "sightingDetails": {
        "DenseNarrativeSection": "string — short omniscient free-form narrative of the event, including any relevant context",
        "trustScore": "integer 0-100 — composite reliability score for the DATA (not the phenomenon): source provenance, completeness of date/time/location/description, investigator class, timeliness, observation modality, witness count and independence",
        "trustScoreNotes": "string — one-sentence justification listing which dimensions drove trustScore",
        "date": "YYYY-MM-DD",
        "timeOfDay": "HH:MM (24-hour format)",
        "location": {
            "name": "string — specific location",
            "latitude": "float — decimal degrees; null if unknown",
            "longitude": "float — decimal degrees; null if unknown"
        },
        "weatherConditions": {
            "generalDescription": "string — description of weather",
            "visibility": "string — Clear, Partially Cloudy, etc.",
            "cloudCover": "string — percentage or description",
            "extremePhenomena": "string — any extreme weather conditions"
        },
        "sightingDuration": "string — duration in hours or minutes",
        "uapCharacteristics": {
            "size": "string — description or comparison",
            "shape": "string — shape of the object",
            "color": "string — colours and patterns",
            "altitude": "string — estimated or measured altitude",
            "speed": "string — estimated or measured speed",
            "movementPattern": "string — zig-zag, straight line, stationary, etc.",
            "changeInAppearance": "string — details of any transformations",
            "sound": "string — description of any associated sounds",
            "electromagneticEffects": "string — disturbances in electronic devices",
            "visualClarity": "string — clarity of the object's appearance",
            "detailedStructure": "string — surface, texture, structural detail",
            "presenceHumanoids": "enum [Y, N, U] — Y ONLY if humanoid beings are described in, on, emerging from, or immediately beside the object itself; ordinary humans (witnesses, bystanders, aircraft pilots, a drone's operator on the ground) never make this Y. N when the source explicitly notes none; U when not mentioned",
            "presenceHumanoids_notes": "string — qualifiers (e.g. 'reported via telepathy', 'in photograph'); empty if none",
            "presenceSounds": "enum [Y, N, U]",
            "presenceLights": "enum [Y, N, U]"
        },
        "observerDetails": {
            "numberOfWitnesses": "integer — exact witness count when reported; null otherwise",
            "numberOfWitnessesFreeform": "string — original phrasing when non-numeric (e.g. 'Multiple', 'Hundreds', 'At least 6')",
            "observerBackground": {
                "occupation": "string — observer occupation",
                "experienceWithAircraft": "string — experience with aircraft or aerial phenomena"
            },
            "psychologicalEffects": "string — psychological/emotional effects on the observer",
            "emotionalStateBeforeSighting": "string — observer's emotional state before the sighting",
            "behavioralResponse": "string — observer's reaction during the sighting",
            "subsequentEffects": "string — physical or psychological effects after the sighting"
        },
        "evidence": [
            {
                "mediaType": "enum [photographic, video, radar_trace, audio, instrumented, physical_sample, sketch]",
                "availability": "enum [available, not_available, attempted_failed, referenced_not_seen, inconclusive]",
                "url": "string — direct link if available, else empty",
                "description": "string — qualifiers (e.g. '8mm film exposed 10 seconds', 'sketch by witness')",
                "chainOfCustody": "string — provenance notes if recorded"
            }
        ],
        "additionalInformation": {
            "interactionWithEnvironment": "string — effects on surroundings",
            "wildlifeBehavior": "string — unusual animal behaviour",
            "previousActivity": "string — history of similar sightings in the area",
            "flightCharacteristics": "string — detailed flight-pattern description",
            "radarData": "enum [available, not_available, available_with_caveat, unknown]",
            "radarDataNotes": "string — qualifiers (e.g. 'tracked by theodolite', 'controller had no contact')",
            "airTrafficCommunications": "string — reports from nearby aircraft or control towers",
            "officialResponses": "string — governmental or scientific responses",
            "historicalComparisons": "string — reference to similar historical sightings",
            "communicationAttempts": "string — perceived attempts at communication from the UAP",
            "temporalAnomalies": "string — reports of lost time or time discrepancies",
            "anomalyDuration": "string — specific times for changes in UAP appearance/behaviour",
            "sightingFrequency": "string — frequency of sightings by the observer"
        }
    },

    # ── Source (parser-emitted; expanded provenance + adjudication) ────────────
    "source": {
        "name": "string — reporting database, publication or organisation (e.g. NICAP, MUFON, AFOSI)",
        "ref": "string — specific citation, case ID or URL for this case",
        "duplicate": "string — case-ref of the original if this is a known duplicate, else blank",
        "documentId": "string — identifier of the upstream source document; copy if visible, else blank",
        "pages": "string — page range within the source document (e.g. 'page_0004-page_0006'); blank if unknown",
        "chunkCount": "integer — number of text chunks contributing to this row; null if unknown",
        "parseWarning": "boolean — true if a parse error was flagged on this row; default false",
        "rawText": "string — un-truncated narrative text of the source",
        "agency": "string — coarse agency taxonomy inferred from the document (FBI, NARA-CIA, DOD, MISC, NASA); null if not indicated",
        "collection": "string — sub-collection within the agency (e.g. 'series-38', 'hs1-834228961'); null if not indicated",
        "tier": "enum [primary_investigative, secondary_investigative, reference_book, press, retrospective_personal_account, unknown] — coarse reliability tier; default 'unknown'",
        "firstRecordedDate": "YYYY-MM-DD — earliest dateline visible in the document text (cable/letter/memo date); null if not visible",
        "contemporaneous": "enum [Y, N, U] — Y if firstRecordedDate is within one year of the sighting, N if later, U if either date is unknown",
        "originalAdjudication": "string — verbatim conclusion by the investigating authority as quoted in the document; null if no adjudication appears",
        "originalAdjudicationType": "enum [unidentified, explained, dismissed, no_investigation, not_stated] — coded reading of the adjudication; use 'not_stated' when the document is silent"
    },

    # ── Manifest (DOWNSTREAM JOIN ONLY — leave every field null) ───────────────
    "manifest": {
        "pdfUrl": "null — populated by manifest join, never by the parser",
        "releaseDate": "null — populated by manifest join, never by the parser",
        "isRedacted": "null — populated by manifest join, never by the parser",
        "documentType": "null — populated by manifest join, never by the parser",
        "documentDescription": "null — populated by manifest join, never by the parser",
        "agencyRelease": "null — populated by manifest join, never by the parser",
        "videoId": "null — populated by manifest join, never by the parser",
        "modalImage": "null — populated by manifest join, never by the parser",
        "joinStatus": "null — populated by manifest join, never by the parser"
    },

    # ── Assessment (parse-time judgement + UFOCAT-style explanation taxonomy) ──
    "assessment": {
        "contradictsUap": "boolean — true if the source itself states the incident was NOT a UAP (film defect, balloon, weather, hoax, sarcastic/diplomatic usage)",
        "notes": "string — short justification for contradictsUap, or any other parse-time annotation",
        "explanationCategory": "enum [conventional_tech, weather, celestial, hoax, misidentification, redacted_explained, unknown] — populated only when contradictsUap is true; sub-classifies the type of conventional explanation given by the source. Use 'unknown' when contradictsUap is true but the source does not specify the explanation type."
    },

    # ── Anomaly (SCU Criterion 3 + UFOSETI Hynek-Vallée extensions) ────────────
    "anomaly": {
        "structure": "boolean — anomalous shape, size, surface or detailed structure",
        "flight": "boolean — anomalous flight (hypersonic without sonic boom, instantaneous acceleration, trans-medium, etc.)",
        "occupant": "boolean — occupants observed or encountered",
        "signal": "boolean — anomalous electronic transmissions or radio behaviour",
        "summary": "string — one-sentence justification for any of the above being true",
        "validated": "enum [Y, N, U] — Y if an official investigative body (USAF, FBI, AARO, NICAP, etc.) reviewed the case and could not produce a conventional explanation; N if the same body produced a conventional explanation; U if no official review conclusion is stated in the source",
        "materialRecovery": "enum [Y, N, U] — Y if the source describes or alleges physical recovery of craft debris, material fragments, or other non-biological physical evidence attributable to the UAP; N if explicitly none; U if unknown or not mentioned",
        "biologicalEntity": "enum [Y, N, U] — Y if the source describes or alleges recovery, encounter, or direct contact with a non-human biological entity (living or deceased); N if explicitly none; U if unknown or not mentioned"
    },

    # ── Date / Time ────────────────────────────────────────────────────────────
    "date_time": {
        "year": "integer YYYY — calendar year; null if unknown",
        "month": "integer 1-12 — calendar month; null if unknown",
        "day": "integer 1-31 — calendar day; null if unknown",
        "day_night": "enum [D, N, U]",
        "local_time": "string HH:MM — local 24-hour start time; empty if unknown",
        "local_time_code": "string — IANA timezone abbreviation or UTC offset (e.g. EST, UTC+2)",
        "duration_min": "integer — total observed duration in minutes; null if unknown",
        "gmt_start": "string HH:MM — start time converted to GMT/UTC; empty if unknown",
        "description": "string — free-text date/time context when exact values cannot be determined",
        "yearUncertainty": "enum [exact, approximate_year, approximate_decade, unknown]",
        "dateUncertainty": "enum [exact, approximate_month, approximate_year, unknown]",
        "inWindow_1945_1975": "boolean — true if the sighting falls within the SCU master study window"
    },

    # ── Location ───────────────────────────────────────────────────────────────
    "location": {
        "country": "string — ISO 3166-1 alpha-2 code (strict). For non-country locations use sentinels: INTL_WATERS, MOON, AQ, MULTIPLE, SCAND, UNKNOWN",
        "state": "string — US Census two-letter code for US states; ISO 3166-2 subdivision code elsewhere; empty if not applicable",
        "city": "string — city, town or nearest known populated place",
        "latitude": "float — decimal degrees WGS-84, positive = North; null if unknown",
        "longitude": "float — decimal degrees WGS-84, positive = East; null if unknown",
        "description": "string — additional location context: landmarks, base name, access restrictions",
        "type": "enum [Atomic site, Military base, Population centre, Waste facility, Water body, River, Airport, Forest, Desert, Ocean, Space flight, Mixed, Other, Rural, Missile testing, Unknown] — primary character of the location",
        "secondaryType": "enum [same values as type, or empty] — for locations that genuinely span two types"
    },

    # ── Witness (structured roles array) ───────────────────────────────────────
    "witness": {
        "count": "integer — exact number of independent witnesses; null if unknown",
        "countFreeform": "string — original phrasing when non-numeric (e.g. 'Multiple', 'Hundreds')",
        "roles": ["enum [Military, Public, Police, Pilot, Civilian, Scientist, Intelligence, Astronaut, Politician, Security] — JSON array of witness roles; no duplicates, order irrelevant. Roles describe the human WITNESSES themselves ('Pilot' = a pilot witnessed the event), never the object's or a drone's operator"],
        "notes": "string — rank/role qualifiers that don't belong in the roles enum (e.g. 'Naval Reserve')",
        "description": "string — broader witness background, credibility assessment, observer-location detail"
    },

    # ── Investigation (SCU Criterion 1 + 4) ────────────────────────────────────
    "investigation": {
        "source": "string — organisation or individual that investigated",
        "investigatorClass": "enum [USAF, NICAP, MUFON, FBI, CIA, NASA, AARO, law_enforcement, civilian_research, press_only, none, other]",
        "startDate": "YYYY-MM-DD — when the investigation began; null if unknown",
        "lagDays": "integer — days between sighting and start of investigation; null if unknown",
        "timeliness": "enum [timely, late, unknown] — timely = lagDays <= 365 (SCU Criterion 1); late = lagDays > 365",
        "reports_within_1_month_of_sighting": "enum [Y, N, U] — Y if the report or field investigation was initiated within one month of the observation; finer-grained than the SCU Criterion 1 one-year rule",
        "reports_within_1_year_of_sighting": "enum [Y, N, U] — Y if the report or field investigation was initiated within one year of the observation (SCU Criterion 1, the one-year timeliness rule)",
        "description": "string — summary of the investigation process and conclusions"
    },

    # ── Craft ──────────────────────────────────────────────────────────────────
    "craft": {
        "primary_shape": "enum [Sphere, Disc, Cigar, Triangle, Diamond, Cylinder, Boomerang, Chevron, Egg, Cone, Saturn, Cross, Rectangle, Teardrop, Fireball, Light, Orb, Tic-Tac, Cube, Pyramid, Oval, Dome, Crescent, Unknown, Other] — dominant observed shape; fold Round/Circle/Ball/Spherical into Sphere. Purely descriptive of the reported appearance, independent of identification (a balloon seen as a sphere is Sphere; the prosaic explanation belongs in assessment.contradictsUap). If the source names a conventional craft (airplane, helicopter, quadcopter drone) without describing one of the listed geometries, use Other — never map a named conventional craft to Disc/Cigar/etc. by analogy",
        "secondary_shape": "enum [same values as primary_shape, or empty]",
        "colour": "string — colour or colour combination observed",
        "size": "enum [Tiny (<0.5 m), Small (0.5-3 m), Medium (3-10 m), Large (10-50 m), Very large (50-200 m), Massive (>200 m), Unknown]",
        "sizeRangeMeters": "string — explicit metre range derived from the size band (e.g. '3-10 m'); empty if Unknown",
        "description": "string — free-text shape, texture, surface markings, lighting and structural detail"
    },

    # ── Performance / 5 Observables (tri-state) ────────────────────────────────
    "performance": {
        "speed_mph": "float — estimated or reported speed in mph; null if unknown",
        "acceleration_g": "float — observed peak acceleration in g; null if unknown",
        "hypersonic": "enum [Y, N, U] — Y ONLY if speed above ~Mach 5 is reported or clearly implied (typically without sonic boom); N if observed speeds were conventional; U if not assessed",
        "instantaneous_acceleration": "enum [Y, N, U] — Y ONLY if the object accelerated from rest or slow motion to extreme speed near-instantaneously, beyond any known aircraft; a bird, insect or drone darting at close range is N; U if not assessed",
        "low_observability": "enum [Y, N, U] — Y ONLY if the object evaded detection when detection was expected (no radar return despite visual contact, abrupt vanishing, cloaking-like behaviour); an object merely too small or too distant for radar is N; U if not assessed",
        "trans_medium_travel": "enum [Y, N, U] — Y ONLY if the object crossed between media (air/water/space) without apparent performance loss; a diving bird or a missile splashing down is N; U if no crossing was observed",
        "positive_lift": "enum [Y, N, U] — Y ONLY if the object holds or gains altitude with NO apparent means of lift or propulsion (no wings, rotors, jets, or balloon envelope). Any aircraft, drone or balloon flying normally is N — merely being airborne or hovering is NOT positive lift. U when the lift mechanism cannot be judged from the source"
    },

    # ── Military context ───────────────────────────────────────────────────────
    "military": {
        "reported_by": ["enum [Military, Public, Police, Pilot, Intelligence] — JSON array of who officially reported the case"],
        "military_public": "enum [Military, Public, Mixed, None] — use None when the military/public character of the location is unknown",
        "facility_name": "string — name of the military base, nuclear site or government facility; empty if not applicable",
        "facility_type": "enum [Aircraft testing, Atomic weapons assembly facilities, Atomic weapons deployment, Atomic weapons stockpile sites, Facility type unknown, ICBM Minuteman III MRV, ICBM Sites, Materials Production, Military base, Missile testing, Other, Population centre, Radioactive materials production plants, Reactor Facility, Space flight, Weapons Deployment Base, Weapons Design & Assembly, Weapons Stockpile, Airport, Unknown, None] — use None when the facility type is unknown",
        "comments": "string — source notes on the military relevance or sensitivity of the case"
    },

    # ── Effects (tri-state + medical-injury extension) ─────────────────────────
    "effects": {
        "atomic_related": "enum [Y, N, U] — Y if the event involved an atomic/nuclear site, material or weapon",
        "communication": "enum [Y, N, U] — Y if communication interference or anomalous signals were reported",
        "physical_effects": "enum [Y, N, U] — Y if physical effects on people, animals, vehicles or environment were reported",
        "text": "string — free-text description of communication anomalies and physical effects",
        "humanInjury": "enum [Y, N, U] — Y if physical injury to human witnesses was explicitly reported (burns, radiation sickness, temporary blindness, trauma, neurological effects, etc.); N if the source explicitly states no injury; U if not mentioned. Do NOT infer from 'physical_effects' alone — requires direct textual evidence of human bodily harm.",
        "humanInjuryNotes": "string — type and severity of injury as described in the source (e.g. 'radiation burns to hands and face', 'temporary paralysis lasting 3 days'); empty if humanInjury is N or U"
    },

    # ── Engagement flags (tri-state observed-fact flags) ───────────────────────
    "engagement_flags": {
        "aircraft_engagement": "enum [Y, N, U]",
        "aircraft_encounters": "enum [Y, N, U]",
        "active_radar_jamming": "enum [Y, N, U]",
        "over_military_installation": "enum [Y, N, U]",
        "during_missile_test": "enum [Y, N, U]",
        "radar_tracking": "enum [Y, N, U]",
        "radio_interference": "enum [Y, N, U]",
        "radar_jamming": "enum [Y, N, U]",
        "directed_radar": "enum [Y, N, U]",
        "coded_radar": "enum [Y, N, U]",
        "multiple_interactive_flight": "enum [Y, N, U]"
    },

    # ── Engagement type (exactly one P; 0-many S; blank otherwise) ─────────────
    "engagement_type": {
        "interactive_flight": "P, S, or blank — UAP flight appeared to react to or interact with observers or aircraft",
        "radical_flight": "P, S, or blank — physically impossible manoeuvres (right-angle turns, instant acceleration, sudden reversal)",
        "loitering": "P, S, or blank — UAP hovered, circled or loitered for an extended period",
        "electronic_transmissions": "P, S, or blank — electronic signals attributed to the UAP were detected",
        "interference_weapons": "P, S, or blank — weapons systems were interfered with or disabled",
        "military_intrusions": "P, S, or blank — UAP intruded into a secured installation or restricted airspace",
        "occupant_encounter": "P, S, or blank — occupants OF THE OBJECT were directly encountered by witnesses. 'Occupant' means a being belonging to the object; a drone operator, conventional-aircraft pilot or ground crew is NOT an occupant — leave blank for those",
        "occupant_observed": "P, S, or blank — occupants OF THE OBJECT were visually observed from a distance, no encounter. Same rule as occupant_encounter: drone operators, aircraft pilots and ground personnel are NOT occupants — leave blank for those",
        "close_approach": "P, S, or blank — UAP made a close approach (<30 m) to witnesses, vehicles or structures",
        "no_engagement": "P, S, or blank — set to P only when no other engagement type applies"
    },

    # ── Case narrative ─────────────────────────────────────────────────────────
    "case_text": {
        "text": "string — faithfully summarised case narrative; the verbatim full text lives in source.rawText"
    }
}

# ---------------------------------------------------------------------------
# FORMAT_SCU_V1 — "SCU v1" parsing schema (compact SCU schema)
#
# The same field set as FORMAT_SCU_V3, minus two blocks:
#   • sightingDetails — the verbose nested narrative group
#   • manifest        — the all-null downstream-join placeholder columns the
#                       parser must always leave null
# Everything else (source, assessment, anomaly, date_time, location, witness,
# investigation, craft, performance, military, effects, engagement_flags,
# engagement_type, case_text) is inherited unchanged. Derived from
# FORMAT_SCU_V3 so it never drifts out of sync; _deep_merge treats schemas as
# read-only, so the shared nested references are safe.
# ---------------------------------------------------------------------------
_SCU_V1_DROP = ("sightingDetails", "manifest")
FORMAT_SCU_V1 = {k: v for k, v in FORMAT_SCU_V3.items() if k not in _SCU_V1_DROP}

# ---------------------------------------------------------------------------
# FORMAT_MINI_SCU — the "tunnel" funnel schema: the ABSOLUTE MINIMUM field set
# that scu_normalizer's five-criterion eligibility gate reads. Every leaf here
# is consumed by `scu_eligible` (or the criteria surface it exposes); nothing
# else is. Purpose: a cheap first pass at scale — parse the whole corpus with
# this ~27-leaf schema, keep only the rows that clear the gate, then RE-PARSE
# just the survivors with the full FORMAT_SCU_V3 / FORMAT_MASTER_SCU_V1.
#
# Derived from FORMAT_SCU_V3 so the field DEFINITIONS (enum lists, operational
# defs, P/S/blank engagement semantics) are byte-for-byte identical — extraction
# quality on these fields does not degrade, only the column count shrinks. It
# uses SCU_v3's native flat paths, so the normalizer reads it with NO auto-map.
#
# Gate provenance (scu_normalizer.py -> scu_eligible), field by field:
#   in_scu_window                  <- date_time.year
#   has_core_fields                <- date_time.{year,month,day} + location.country
#   day_night_resolved             <- date_time.day_night
#   has_investigation_channel      <- investigation.source
#   (timeliness_status surface)    <- investigation.reports_within_1_{month,year}...
#   has_credible_witness           <- witness.roles
#   has_anomalous_characterization <- craft.primary_shape + performance.* (5)
#                                     + engagement_type.occupant_{observed,encounter}
#   has_engagement_signal          <- engagement_type.* (9 activity tags)
#   military_public_known          <- military.military_public
#   contradicts_uap (exclusion)    <- assessment.contradictsUap
# Any change to the gate's inputs must be mirrored here; tests/test_scu.py
# asserts MINI covers every gate input and stays a strict subset of SCU_v3.
# ---------------------------------------------------------------------------
def _build_mini_scu(v3):
    def keep(block, *keys):
        return {k: v3[block][k] for k in keys}
    return {
        "date_time": keep("date_time", "year", "month", "day", "day_night"),
        "location": keep("location", "country"),
        "investigation": keep(
            "investigation", "source",
            "reports_within_1_month_of_sighting",
            "reports_within_1_year_of_sighting"),
        "witness": keep("witness", "roles"),
        "craft": keep("craft", "primary_shape"),
        "performance": keep(
            "performance", "hypersonic", "instantaneous_acceleration",
            "low_observability", "trans_medium_travel", "positive_lift"),
        # all 10 engagement tags: 9 activity signals + the no_engagement guard
        # (kept LAST, as generated, so the autoregressive order holds)
        "engagement_type": dict(v3["engagement_type"]),
        "military": keep("military", "military_public"),
        "assessment": keep("assessment", "contradictsUap", "notes"),
    }

FORMAT_MINI_SCU = _build_mini_scu(FORMAT_SCU_V3)

# ---------------------------------------------------------------------------
# FORMAT_MASTER_SCU_V1 — "MasterSCU v1" parsing schema (the default)
#
# The full canonical SCU master schema, loaded from uap_master_schema.json so
# the field set stays in sync with that single source of truth. It is the most
# complete schema in the app: record/study provenance, source, date_time,
# location, witness, detection, object, behavior(+performance), anomaly,
# engagement(types/flags), military, effects, entities(+morphology/tools),
# contact, environment, evidence, classification, assessment, scenarios,
# investigation, narrative and context. Falls back to FORMAT_SCU_V3 if the JSON
# file is unavailable, so the app always starts.
# ---------------------------------------------------------------------------
import json as _json
from pathlib import Path as _Path
try:
    FORMAT_MASTER_SCU_V1 = _json.loads(
        (_Path(__file__).resolve().parent / "uap_master_schema.json")
        .read_text(encoding="utf-8")
    )
except Exception:
    FORMAT_MASTER_SCU_V1 = FORMAT_SCU_V3

# ---------------------------------------------------------------------------
# Merged compact schema — used by the Markdown-folder ingestion agent.
# Covers the essential fields from both FORMAT_LONG (sightingDetails) and
# FORMAT_LONG_XLSX (SCU groups) so the result feeds either downstream pipeline.
# The LLM is instructed to be concise: fill what is explicitly stated, leave
# everything else null / blank.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# UFOSETI (RU) format — Russian UFO/UAP database schema
# Column names mirror the UFOSETI database export; ar_code values follow the
# UFOCAT Codebook 2023: https://cufos.org/PDFs/UFOCAT%20Codebook%202023.pdf
# ---------------------------------------------------------------------------
FORMAT_UFOSETI_RU = {
    # 15 columns matching the UFOSETI export schema (final_ufoseti_dataset.xlsx, 3 069 rows)
    "obs_number":         "integer — UFOSETI internal observation ID; preserve the raw number exactly (e.g. 3776)",

    "en_description":     "string — English-language narrative description of the sighting event",
    "ru_description":     "string — Russian-language narrative description of the same event (Описание на русском языке)",

    "country":            "string — Country where the sighting occurred (e.g. Russia, Kazakhstan, Ukraine, United States)",

    # ar_code: two-character UFOCAT Anomaly Rating code from the UFOCAT Codebook 2023
    #   A digit = Anomaly level:  0 = Not Anomalous | 1 = Probably Not Anomalous
    #                             2 = Probably Anomalous | 3 = Anomalous
    #   R digit = Reliability:    0 = Hoax or Mistake | 1 = Rumor / Media Publication
    #                             2 = Reliable Source / Primary Source | 3 = Indirect Testimony
    #                             4 = Indirect Evidence | 5 = Have Reliable Data
    #                             6 = Have Irrefutable Public Data
    "ar_code":            "string — UFOCAT two-part Anomaly-Reliability code, format A{0-3}R{0-6}. "
                          "A = anomaly level (0 Not Anomalous → 3 Anomalous); "
                          "R = evidence reliability (0 Hoax/Mistake → 6 Irrefutable Public Data). "
                          "E.g. A2R2 = Probably Anomalous | Reliable Source. Leave blank if unknown.",

    # Hynek-Vallée phenomenological classification used by UFOSETI
    #   Category prefix: FB = Fly-By (no maneuver noted)
    #                    MA = Maneuver (observed flight maneuver)
    #                    AN = Anomalous phenomenon (non-craft anomaly)
    #                    CE = Close Encounter (object within ~150 m)
    #   Subtype digit:   1 = no physical effects or traces
    #                    2 = physical / electromagnetic effects
    #                    3 = entity or occupant observed
    #                    4 = abduction or reality transformation (AN4)
    #                    5 = fatal outcome or bilateral contact (CE5)
    "hynek_vallee_code":  "string — Hynek-Vallée classification: prefix (FB Fly-By, MA Maneuver, AN Anomaly, CE Close Encounter) "
                          "plus subtype 1-5 (1 no effects; 2 physical/EM effects; 3 entity observed; 4 abduction/transformation; 5 fatal/contact). "
                          "Observed values: FB1, FB2, FB4, FB5, MA1, MA2, MA3, MA4, MA5, AN1, AN2, AN3, AN4, AN5, CE1, CE2, CE3, CE4, CE5. "
                          "Leave blank if unknown.",

    "authenticities":     "string — UFOSETI investigation status. Allowed values: "
                          "Unconfirmed (report received, not yet reviewed), "
                          "Under investigation (active review in progress), "
                          "Investigation was made (review completed), "
                          "Investigation suspended (review halted, no conclusion), "
                          "Insufficient data (not enough information to assess), "
                          "Repetition (duplicate or repeat of a known sighting). "
                          "Leave blank if unknown.",

    "observation_date":   "string — Date and time of the sighting in DD-MM-YYYY HH:MMZ format (UTC), e.g. 18-05-2024 07:28Z",
    "report_date":        "string — Date and time the report was submitted in DD-MM-YYYY HH:MMZ format (UTC), e.g. 20-05-2024 13:13Z",

    "shape":              "string — Shape of the observed object. Allowed values: "
                          "Round, Circular, Spherical, Triangular, Star-shaped, Cigar-shaped, Oval, Hemisphere, Polygonal. "
                          "Leave blank if unknown.",

    "observer_direction": "string — Compass direction the observer was facing during the sighting. Allowed values: "
                          "North, North-East, East, South-East, South, South-West, West, North-West, No direction. "
                          "Leave blank if unknown.",

    # ar_code_full is the human-readable expansion of ar_code; the two must always agree
    "ar_code_full":       "string — Full human-readable UFOCAT Anomaly Rating: {Anomaly level} | {Reliability level}. "
                          "Anomaly levels: Not Anomalous | Probably Not Anomalous | Probably Anomalous | Anomalous. "
                          "Reliability levels: Hoax/Mistake | Rumor/Media Publication | Reliable Source/Primary Source | "
                          "Indirect Testimony | Indirect Evidence | Have Reliable Data | Have Irrefutable Public Data. "
                          "Must be consistent with ar_code. E.g. A2R2 → Probably Anomalous | Reliable Source/Primary Source.",

    "source":             "string — Originating source: direct URL (YouTube, VK, NUFORC, etc.) or organisation name "
                          "(e.g. MUFON — Mutual UFO Network, NUFORC — National UFO Reporting Center); leave blank if unknown",

    "ufoseti_report_date":"string — Timestamp when this record was ingested into the UFOSETI database, "
                          "format YYYY-MM-DD HH:MM:SS UTC, e.g. 2024-05-20 13:13:03 UTC",

    "url":                "string — Direct URL to the original UFOSETI report page, "
                          "e.g. https://ufoseti.org/ru/reports/3776-inoy-nlo-russian-federation-chelyabinskaya-oblast"
}

FORMAT_MERGED = {
    # ── Default UAP format fields ──────────────────────────────────────────────
    "sightingDetails": {
        "DenseNarrativeSection": "2-3 sentence omniscient summary of the event",
        "trustScore":            "integer 0-100 reliability estimate",
        "date":                  "YYYY-MM-DD or partial (YYYY-MM or YYYY)",
        "location": {
            "name":      "most specific known place name",
            "latitude":  "decimal degrees or null",
            "longitude": "decimal degrees or null"
        },
        "uapCharacteristics": {
            "shape":           "dominant shape or null",
            "color":           "colour description or null",
            "size":            "size description or null",
            "speed":           "speed description or null",
            "movementPattern": "movement description or null",
            "sound":           "sound description or null",
            "presenceLights":  "Yes / No / null"
        }
    },
    # ── SCU spreadsheet format fields ─────────────────────────────────────────
    "source": {
        "name": "reporting database, publication or organisation",
        "ref":  "case ID, document reference or URL"
    },
    "date_time": {
        "year":         "integer YYYY or null",
        "month":        "integer 1-12 or null",
        "day":          "integer 1-31 or null",
        "day_night":    "D / N / U",
        "duration_min": "integer minutes or null"
    },
    "location": {
        "country":  "ISO country name or code",
        "state":    "state / province or null",
        "city":     "nearest city or null",
        "latitude":  "float or null",
        "longitude": "float or null",
        "type":      "enum [Atomic site, Military base, Population centre, Water body, Airport, Forest, Desert, Ocean, Unknown]"
    },
    "witness": {
        "count": "integer or null",
        "type":  "comma-separated subset of [Military, Public, Police, Pilot, Civilian, Scientist]"
    },
    "craft": {
        "primary_shape": "enum [Sphere, Disc, Cigar, Triangle, Diamond, Cylinder, Boomerang, Egg, Cone, Fireball, Light, Orb, Tic-Tac, Rectangle, Unknown, Other]",
        "colour":        "colour string or null",
        "size":          "enum [Tiny, Small, Medium, Large, Very large, Massive, Unknown]"
    },
    "performance": {
        "hypersonic":                 "Y or blank",
        "instantaneous_acceleration": "Y or blank",
        "low_observability":          "Y or blank",
        "trans_medium_travel":        "Y or blank",
        "positive_lift":              "Y or blank"
    },
    "military": {
        "reported_by":   "comma-separated subset of [Military, Public, Police, Pilot, Intelligence]",
        "facility_name": "name of base / site or null",
        "facility_type": "SCU facility enum or null"
    },
    "effects": {
        "atomic_related":   "Y / N",
        "physical_effects": "Y / N"
    },
    "engagement_type": {
        "interactive_flight":       "P, S, or blank",
        "radical_flight":           "P, S, or blank",
        "loitering":                "P, S, or blank",
        "close_approach":           "P, S, or blank",
        "no_engagement":            "P, S, or blank"
    },
    "case_text": {
        "text": "verbatim or faithfully summarised case narrative"
    }
}

# ── FORMAT_NUFORC ──────────────────────────────────────────────────────────────
# US National UFO Reporting Center | ~70,000 rows | EN | NUFORC 2023.03.csv
FORMAT_NUFORC = {
    "id":          "string — unique NUFORC case ID (e.g. S177105)",
    "city":        "string — city or town of sighting (uppercase)",
    "district":    "string — state, province or region (uppercase)",
    "country":     "string — ISO 3166-1 alpha-2 country code (e.g. US, GB)",
    "water":       "string — associated water body; null if unknown",
    "other":       "string — additional location qualifier; null if unknown",
    "date":        "string — ISO 8601 datetime of sighting (e.g. 2023-07-29 17:45:00)",
    "date_detail": "string — human-readable date/time as submitted (e.g. 07/29/23 5:45 PM)",
    "description": "string — free-text witness narrative",
}

# ── FORMAT_BLUE_BOOK ───────────────────────────────────────────────────────────
# US Air Force Project Blue Book (1947–1969) | ~15,000 rows | EN | bluebook1.csv + bluebook2.csv
FORMAT_BLUE_BOOK = {
    "case_id":         "string — Blue Book case identifier (e.g. BB-8371)",
    "date":            "string — date of the sighting (various formats); null if unknown",
    "location":        "string — free-text location (city, state or country)",
    "state":           "string — US state abbreviation (e.g. NY, CA); null for non-US",
    "country":         "enum [US, EU, P, AS, CN, A, CA, SA, AF, AU, ME, JP, GB, M, OC, AN] — country or continental code; null if unknown",
    "incident":        "string — free-text summary of the incident",
    "special_comments":"string — investigator annotations; null if unknown",
    "shape":           "string — observed shape (free-text; ~400 unique values; e.g. Disc, Cigar, Light, Round, Ball, DomeDisc, Ovoid, Oval, Cylinder, Sphere, Triangle, Fireball); null if unknown",
    "size_meters":     "float — estimated physical size in metres; null if unknown",
    "result":          "enum [Meteor, Aircraft, Balloon, Star, Venus, Satellite, Reflection, Debris, Mars, Mirage, Rocket, Hoax, Jupiter, Flare, Re-entry, Unidentified, Insufficient Data, Unknown, Psychological, Natural, Moon, Radar, Light, Astronomical, Other] — official investigation conclusion; null if unknown",
}

# ── FORMAT_UK_NATIONAL_ARCHIVES ────────────────────────────────────────────────
# UK MoD / National Archives UAP catalogue (1978–2000) | ~2,827 rows | EN | uk.csv
FORMAT_UK_NATIONAL_ARCHIVES = {
    "date":               "string — date of sighting (YYYY-MM-DD)",
    "catalog_entry":      "string — UK National Archives DEFE reference (e.g. DEFE-24-1939-13)",
    "location":           "string — free-text city, county and country (e.g. Laughton, Sheffield, UK)",
    "summary_description":"string — free-text summary of the sighting as filed by the MoD",
    "shape":              "string — observed shape (free-text; ~936 unique values; major: light, lights, round, circular, star, triangular, cigar, oval, saucer, ball, disc, sphere, oblong); null if unknown",
    "size":               "string — size descriptor (free-text; major: large, small, very large, aircraft size, football, moon); null if unknown",
    "special_comments":   "string — supplementary MoD analyst notes, identification attempts and cross-references; null if unknown",
}

# ── FORMAT_COBEPS_NOTIFICATIONS_PAN ───────────────────────────────────────────
# COBEPS Belgium official PAN notifications up to 31 Dec 2023 | ~1,308 rows | FR | NOTIFICATIONS PAN (Belgique) 31_12_2023.csv
FORMAT_COBEPS_NOTIFICATIONS_PAN = {
    # Administrative
    "NO":           "integer — sequential record number",
    "N_Cobeps":     "string — internal COBEPS case reference (N°Cobeps)",
    # Temporal
    "CommAn":       "integer — year report was received by COBEPS (2006–2023)",
    "Origininfo":   "enum [COBEPS, REUB, MUFON, ovni-ufologie.com, Les ovnis Forum ufologique, Belgisch UFO-Meldpunt, SOS-OVNI, AREPS, Forum/blog internet belge, Forum/blog internet étranger, GERU] — channel through which report was received",
    "ExactMom":     "enum [TRUE, FALSE] — whether the observation time is exact",
    "J_N_C":        "enum [J, N, C] — time of day: J=day (jour), N=night, C=twilight/crepuscular (J/N/C column)",
    "AnObs":        "integer — year of observation",
    "MoisObs":      "integer 1–12 — month of observation",
    "JourObs":      "integer 1–31 — day of month of observation",
    "JourSem":      "integer 1–7 — day of week (1=Mon, 7=Sun)",
    "HeurObs":      "integer 0–24 — hour of observation",
    "DecaGMT":      "integer — UTC offset in hours (1 or 2)",
    "GMT":          "string — UTC time of sighting (HH:MM)",
    "Duree":        "float — duration in minutes; null if unknown",
    "Secondes":     "integer — duration in seconds; null if unknown",
    "Multi":        "enum [0, 1] — multi-event flag: 0=single event, 1=part of multi-event series",
    # Location
    "Lieu":         "string — locality name (Belgium)",
    "Nuts2":        "enum [Hainaut, Liège, Namur, Brabant Wallon, Bruxelles, Luxembourg, Vlaams-Brabant, West-Vlaanderen, Oost-Vlaanderen, Antwerpen, Limburg] — Belgian NUTS-2 province",
    "Lat":          "float — latitude in decimal degrees WGS-84; null if unknown",
    "Long":         "float — longitude in decimal degrees WGS-84; null if unknown",
    # Witness
    "NbTem":        "integer — number of witnesses",
    "EnMouv":       "enum [0, 1] — witness in motion: 0=stationary, 1=moving",
    "Source":       "enum [1, 2, 3] — source quality: 1=direct, 2=secondary, 3=press/media",
    # Phenomenology
    "Texte":        "string — free-text case narrative",
    "NbPhe":        "integer — number of phenomena/objects observed",
    "Dessinable":   "enum [0, 1] — phenomenon describable enough to sketch: 0=no, 1=yes",
    "Symetrie":     "string — symmetry code (Fx=frontal, Sx=sagittal; digit=planes of symmetry; e.g. F0, S1, S3)",
    "CouleurUni":   "string — dominant single colour (free-text); null if unknown",
    "CouleurForm":  "string — secondary/form colour (free-text); null if unknown",
    "Bruit":        "enum [0, 1] — sound reported: 0=no, 1=yes",
    "Deplacement":  "enum [Rectiligne ou courbe mais continu, Immobile, Complexe: plusieurs types de mouvements différents, Immobile puis en mouvement rectiligne, Changements brutaux de direction et/ou de vitesse, Rectiligne ou en larges courbes, Globalement rectiligne mais saccades et/ou légères variations de vitesse, Complexe et difficilement descriptible, 1 à 3 changements importants de directions et de vitesses, Sacades et légères variations de vitesse] — movement pattern; null if unknown",
    "AzimutDeb":    "float — starting azimuth in degrees (0–360); null if unknown",
    "ElevDeb":      "float — starting elevation angle in degrees (0–90); null if unknown",
    "AzimutFin":    "float — ending azimuth in degrees; null if unknown",
    "ElevFin":      "float — ending elevation angle in degrees; null if unknown",
    "Photo":        "integer — number of photographs (0 if none); null if unknown",
    "Video":        "integer — number of video recordings (0 if none); null if unknown",
    "Trace":        "enum [0, 1] — physical trace left: 0=no, 1=yes",
    # Classification
    "OrganTrait":   "string — organisation that processed the case (e.g. COBEPS, REUB)",
    "EtapEnq":      "enum [Enquête achevée, Clôturé sans enquête, Pas d'information, Evaluation en cours, Enquête en cours] — investigation stage",
    "Rapport":      "enum [FI, RDE, CRE] — report type produced; null if none",
    "GEIPAN":       "enum [a, b, c, d, x, ec] — GEIPAN/PAN classification: a=identified, b=probably identified, c=insufficient data, d=unidentified, x=unclassified, ec=under investigation",
    "Hynek":        "enum [LN, DD, RR1, RR2, RR3, RR4, PS] — Hynek classification; null if unknown",
    "TypeIdent":    "enum [Aérostat, Aérodyne, Astronautique, Astronomique, Rentrée atmosphérique, Projection lumineuse, Animaux volants, Eclairage, Psychologique, Projectiles lumineux] — identified explanation category; null if unidentified",
    "AltitPhen":    "enum [0 Sol, 1 Basses couches, 2 Troposphère, 3 Stratosphère, 4 Mésosphère, 5 Thermosphere, 6 Exosphère, 8 Spatiale] — altitude layer of the phenomenon; null if unknown",
    "AvisTem":      "enum [0, 1] — witness confidence in own report: 0=uncertain, 1=confident; null if unknown",
}

# ── FORMAT_COBEPS_COB_2021 ─────────────────────────────────────────────────────
# COBEPS COB Online historical catalogue (pre-2021) | ~2,111 rows | FR | COB_2021 - COB Online.csv
FORMAT_COBEPS_COB_2021 = {
    "CodeCOB":      "string — unique COB case code",
    "Lien_PDF":     "string — URL to associated PDF report; null if none",
    "Date_Encodage":"string — date the case was encoded",
    # Location
    "Lieu":         "string — locality of the sighting (Belgium)",
    "Province":     "enum [Hainaut, Bruxelles, Liège, Namur, West-Vlaanderen, Antwerpen, Vlaams-Brabant, Luxembourg, Brabant Wallon, Oost-Vlaanderen, Limburg] — Belgian province",
    "Long":         "float — longitude in decimal degrees",
    "Lat":          "float — latitude in decimal degrees",
    # Temporal
    "Date_p":       "string — primary (earliest) date of observation",
    "Heure_p":      "string — primary time of observation (HH:MM)",
    "Imp_heure":    "enum [0, 1] — time imprecision: 0=exact, 1=approximate",
    "Decalage":     "enum [00:00, 01:00, 02:00] — UTC offset string",
    "J_N_C":        "enum [J, N, C, D] — time of day: J=day, N=night, C=twilight, D=dawn",
    # Witness
    "RD1":          "enum [0, 1] — data reliability: 0=lower, 1=higher",
    "Nb_temoins":   "string — number of witnesses (integer or >1)",
    "TM":           "enum [0, 1] — multiple-witness: 0=single, 1=multiple",
    "Etat_Ciel":    "enum [Dégagé, Nuageux, Couvert, Dégagé>Couvert] — sky condition; null if unknown",
    # Phenomenology
    "Description":  "string — free-text narrative of the sighting",
    "Mob":          "enum [0, 1] — witness mobile: 0=stationary, 1=moving",
    "Nb_P":         "string — number of objects/phenomena",
    "Apparence":    "enum [Forme, Lumière, Ligne, Pt] — primary phenomenological appearance: Forme=shaped object, Lumière=light, Ligne=line, Pt=point",
    "Formation":    "enum [Triangulaire, Sans, Inconnue, Complexe, Quadrangulaire, Linéaire, Circulaire, Paire] — formation shape when multiple objects; null if single",
    "Trajectoire":  "enum [Arrêtée, Stationnaire, Rectiligne, Complexe, Saccadée, Brisée, Courbe, Cyclique] — movement trajectory; null if unknown",
    "Duree":        "float — duration in minutes; null if unknown",
    "Photos":       "integer — number of photographs taken; null if none",
    "Niveau_source":"enum [1, 2, 3] — source quality: 1=direct/firsthand, 2=secondary, 3=tertiary",
    "Lune_Pres":    "enum [0, 1] — moon above horizon: 0=no, 1=yes",
    # Ratings
    "CR":           "integer 0–5 — credibility rating",
    "ET":           "integer 0–5 — strangeness/exceptionality rating",
    "Hynek":        "enum [LN, DD, RR1, RR2, RR3, RR4] — Hynek classification",
    "Forme":        "enum [OV, SV, SP, TR, CI, RE, CY, PO, TZ] — shape code: OV=oval, SV=sphere, SP=disc/saucer, TR=triangle, CI=cigar, RE=rectangle, CY=cylinder, PO=polygon, TZ=teardrop; null if unknown",
    "PAN":          "enum [A, B, C, D] — GEIPAN/PAN classification: A=identified, B=probably identified, C=insufficient data, D=unidentified",
    "Identification":"string — free-text identified explanation; null if unidentified",
}

# ── FORMAT_GEP ─────────────────────────────────────────────────────────────────
# GEP Germany UFO/UAP reports 1972–2023 | ~5,577 rows | DE | UFO-UAP-Falldaten_GEP_1972-2023.csv
FORMAT_GEP = {
    "Fallnummer":                              "string — GEP internal case number",
    "Datum":                                   "string — date of observation (YYYY-MM-DD or free-text)",
    "Land":                                    "string — country of sighting (primarily Deutschland)",
    "Bundesland":                              "string — German federal state (Bundesland)",
    "Sichtungsort":                            "string — city or locality of sighting",
    "lokale_Uhrzeit":                          "string — local time of sighting (HH:MM)",
    "Zeitzone":                                "enum [MESZ, MEZ, ESZ, EZ, ES, WESZ, ADT, WEZ, BST, EDT, EST, CET, CEST] — timezone abbreviation",
    "Tageszeit":                               "enum [abends, nachts, tagsüber, nachmittags, mittags, morgens, spätabends, spätnachmittags, vormittags, frühabends] — time-of-day descriptor (German: abends=evening, nachts=night, tagsüber=daytime, morgens=morning, etc.); null if unknown",
    "UTC":                                     "string — UTC datetime of sighting",
    "Erstkontakt":                             "enum [Falldatenbank, E-Mail, Anruf, Kontaktformular, Facebook, Brief, Fragebogen, GEP-Forum, Medien, Sonstiges] — channel of first contact/report submission",
    "Gemeldet":                                "string — date the case was reported",
    "Sachverhalt":                             "string — free-text summary of the sighting (German)",
    "Klassifikation_Hynek":                    "enum [NL, DD, CE I, CE II, CE III, CE IV, CE 4.2, S, Ungenügende Daten] — Hynek classification",
    "Klassifikation_Hendry":                   "enum [IFO, NEAR IFO, PROBLEMATIC UFO, GOOD UFO, Ungenügende Daten, Hoax, S] — Hendry IFO/UFO classification",
    "Klassifikation_IFO_Verifikationsschluessel": "enum [V1, V2, V3] — IFO verification confidence level (V1=high, V3=low)",
    "Identifikation":                          "string — specific identified object or phenomenon (free-text); null if unidentified",
    "Identifikation_Gruppe":                   "enum [Modell-Heißluftballone / Himmelslaternen, Folienballons / Forschungsballons / LED-Ballons / Heliumballons, Flugzeuge / Hubschrauber / Kondensstreifen, Planeten / Sterne / Mond, Satelliten / Flares / Raumstation, Insekten / Vögel / Partikel / Tropfen, Lichteffektgeräte / Skybeamer / Laser / LIDAR / Scheinwerfer / Beleuchtung, Meteore / Feuerkugeln / Re-entries, Drohnen, Atmosphärische Effekte / Wetterphänomene, Unbekannt, Handgemachtes, Ungenügende Daten] — category group for identified objects; null if unidentified",
}

# ── FORMAT_UPDB_NICAP ──────────────────────────────────────────────────────────
# UPDB / NICAP NSID database | ~5,885 rows | EN | NSID_DBListingbyDateLinks.csv
FORMAT_UPDB_NICAP = {
    "Date":           "string — date of sighting (YYMMDD format; 000000=unknown)",
    "City":           "string — city or locality",
    "State_or_Country":"string — US state abbreviation or full country name",
    "Cat":            "enum [1, 2, 3, 4, 5, 6, 7, 8, 9, 11] — NICAP/NSID phenomenological category: 1=Nocturnal lights, 2=Daylight discs, 3=Radar cases, 4=Close encounters (CE I–IV), 5=Photographic cases, 6=Physical traces, 7=Medical effects on witnesses, 8=Animal reactions, 9=Electromagnetic effects, 11=Unclassified/other",
    "BB":             "string — linked Blue Book case ID if cross-referenced; null if none",
    "NC_flag":        "string — NICAP data quality flag (sparse); null if unknown",
    "LC":             "string — location confidence flag (sparse); null if unknown",
    "Description":    "string — free-text case description",
}

# ── FORMAT_OVNIBASE ────────────────────────────────────────────────────────────
# OVNIBASE French UFO database | ~3,145 rows | FR | ovni-base.csv
FORMAT_OVNIBASE = {
    "Num_cas":          "integer — sequential case number",
    "Depart":           "string — French département number (e.g. 51; Départ. column)",
    "Ville":            "string — city or locality",
    "Latitude":         "float — latitude in decimal degrees; null if unknown",
    "Longitude":        "float — longitude in decimal degrees; null if unknown",
    "CR_Observation":   "string — free-text narrative of the sighting (CR Observation column)",
    "Typ_Obs":          "enum [Visuel : Eloigné, Visuel : proche, Visuel proche avec personnages, Seulement entités] — type of observation: distant visual, close visual, close visual with entities, entities only",
    "Date":             "string — date of sighting (DD/MM/YYYY)",
    "Heure":            "string — time of sighting (HH:MM); null if unknown",
    "Duree":            "string — duration (free-text); null if unknown",
    "Nbre_Objets":      "integer — number of objects observed",
    "Type_Objet":       "string — shape/type of object (free-text; ~115 unique values; major: Non-défini, Boule Lumineuse, Disque, Cigare, Sphère, Ovoïde, Triangle, Elliptique, Soucoupe avec dôme, Demi sphère, Rectangle, Cylindre); null if unknown",
    "Couleur":          "enum [Rouge, Blanc, Orange, Métallique (argent,), Jaune, Plusieurs couleurs, Changement couleur, Vert, Gris, Bleu, Noir, Rose] — dominant colour; null if unknown",
    "Brillance":        "enum [Brillant, éblouissant, Halo, fluorescent, Faiblement lumineux, Réfléchit la lumière, Brillant par intermitence, Non lumineux] — luminosity descriptor; null if unknown",
    "Effet_visuel":     "string — visual effect observed (free-text; e.g. Traînée, faisceau de lumière, hublots, feux clignotants, gerbes d'étincelles, Rayon); null if unknown",
    "Disp_inst":        "enum [o, n] — instantaneous disappearance: o=yes, n=no",
    "Vitesse":          "string — speed descriptor (free-text; ~63 unique values; e.g. Rapide, Lent, Immobile, Extrêmement rapide, Immobile puis accélération foudroyante); null if unknown",
    "Mouvement_type_vol":"string — movement/flight type (free-text; ~92 unique values; e.g. En ligne droite, Diverses manœuvres, Au sol puis décollage, Virage à angle droit); null if unknown",
    "Taille":           "string — size descriptor (~195 unique; apparent or estimated; e.g. 2 à 5 mètres (évaluée), Pleine Lune (apparente)); null if unknown",
    "Type_Entite":      "string — entity/occupant type description (~58 unique, sparse); null if none",
    "Nbre_Entite":      "string — number of entities (0 if none); null if unknown",
    "Action_entite":    "string — actions performed by entities (comma-separated, sparse); null if none",
    "Effets_temoin":    "string — physiological/psychological effects on witness (comma-separated; ~115 unique values); null if none",
    "Effet_Physique":   "string — physical effects on environment or vehicles (sparse); null if none",
    "Nbre_Temoins":     "integer — number of witnesses",
    "Enq_Off":          "enum [o, n] — officially investigated: o=yes, n=no",
    "Meteo":            "enum [Beau temps, ciel clair, Nuages épars, Ciel couvert haute altitude, Ciel couvert basse altitude, Pluie continue faible, Pluies éparses, Brouillard, Grosse pluie, Orage, Tempête, Neige] — weather conditions; null if unknown",
    "Annee":            "integer — year of observation",
    "Atter":            "enum [o, n] — landing reported: o=yes, n=no",
}

# ── FORMAT_UFOSETI ──────────────────────────────────────────────────────────────
# UFOSETI.org Russian/international UAP database | ~3,069 rows | EN+RU | final_ufoseti_dataset.csv
# Note: FORMAT_UFOSETI_RU (above) is the LLM extraction template; this mirrors the raw column schema.
FORMAT_UFOSETI = {
    "obs_number":        "integer — unique UFOSETI observation number",
    "en_description":    "string — English-language description of the event",
    "ru_description":    "string — Russian-language description of the event (Описание на русском языке)",
    "country":           "string — country of sighting (full name)",
    "ar_code":           "enum [A0R0, A0R1, A0R2, A1R1, A1R2, A1R3, A2R1, A2R2, A2R3, A2R4, A3R1, A3R2] — UFOCAT two-part Anomaly-Reliability code: A0–A3 = anomaly level (0 Not Anomalous → 3 Anomalous); R0–R4 = source reliability",
    "ar_code_full":      "string — human-readable expansion of ar_code (e.g. Probably Anomalous | Reliable Source/Primary Source)",
    "hynek_vallee_code": "enum [FB1, FB2, FB5, MA1, MA2, AN1, AN2, CE1, CE2, CE3, CE4, CE5] — combined Hynek/Vallée classification; null if unknown",
    "authenticities":    "enum [Unconfirmed, Investigation was made, Under investigation, Insufficient data, Repetition, Investigation suspended] — report authentication status",
    "observation_date":  "string — date and time of observation (DD-MM-YYYY HH:MMZ format, UTC)",
    "report_date":       "string — date and time the report was filed",
    "shape":             "enum [Round, Circular, Spherical, Triangular, Star-shaped, Cigar-shaped, Oval, Hemisphere, Polygonal] — shape of the observed object; null if unknown",
    "observer_direction":"enum [No direction, East, West, South-West, North-West, South, South-East, North, North-East] — cardinal direction the observer was facing",
    "source":            "string — original source of the report (free-text URL or organisation name); null if unknown",
    "ufoseti_report_date":"string — date/time the case was entered into the UFOSETI database (UTC)",
    "url":               "string — URL to the case page on ufoseti.org",
}

# ── FORMAT_UAP_SIGHTINGS_GITHUB ───────────────────────────────────────────────
# MiChaelinzo/UAP_SIGTHINGS multi-source sighting JSON | ~1,000 rows (sample) | EN | getData2.json
FORMAT_UAP_SIGHTINGS_GITHUB = {
    "id":          "string — unique sighting ID",
    "city":        "string — city of sighting",
    "country":     "string — ISO 3166-1 alpha-2 country code (primarily US)",
    "date":        "string — ISO 8601 datetime of sighting",
    "date_detail": "string — human-readable date/time string",
    "description": "string — free-text witness narrative",
    "district":    "string — state, province or region (uppercase)",
    "latitude":    "string — latitude in decimal degrees (sparse); null if unknown",
    "longitude":   "string — longitude in decimal degrees (sparse); null if unknown",
    "other":       "string — additional notes or source-specific data; null if unknown",
    "source":      "enum [NUFORC, UFODNA, NICAP] — originating database",
    "source_id":   "string — case ID within the originating source database",
    "water":       "string — water body reference (sparse); null if unknown",
    "word_count":  "integer — word count of the description field",
}

# ── FORMAT_WEINSTEIN_PILOT_CATALOG ────────────────────────────────────────────
# Dominique Weinstein Pilot Catalog — aircraft crew UAP encounters 1916–1999 | ~1,300 rows | EN | Weinstein UFO Catalog 2001.xlsx
FORMAT_WEINSTEIN_PILOT_CATALOG = {
    "W_num":                   "integer — sequential Weinstein case number (W# column)",
    "DATE":                    "string — date in YY.MM.DD format (e.g. 44.06.00 = June 1944)",
    "TIME":                    "string — local time (HH:MM); null if unknown",
    "ZT_LT":                   "enum [ZT, LT] — time zone indicator: ZT=Zulu/UTC, LT=local time; null if unknown",
    "COUNTRY":                 "string — country or ocean/region (full name)",
    "LOCATION":                "string — specific location, airway or coordinates",
    "LAT_LNG":                 "string — coordinates (sparse, free-text); null if unknown",
    "TYPE_OF_PLANE_AND_WITNESSES": "string — aircraft type and witness description",
    "WHO":                     "string — witness role (e.g. pilot, crew, passengers)",
    "UFO_DESCRIPTION":         "string — free-text description of the UAP",
    "Radar":                   "enum [AR, GR, RO, NR, GR AR, GRAR, GRARO] — radar detection: AR=airborne radar, GR=ground radar, RO=radar only, NR=no radar; null if unknown",
    "G":                       "string — ground witness also saw the object (G if confirmed); null otherwise",
    "X":                       "string — multiple independent witnesses (X if present); null otherwise",
    "E":                       "string — electromagnetic/physical effect on aircraft (E if present); null otherwise",
    "SOURCES":                 "string — source references (coded bibliography)",
}

# ── FORMAT_PETROWITSCH_LATAM ──────────────────────────────────────────────────
# Petrowitsch Catalogue — Chilean & Latin American UAP cases | ~3,529+ rows | ES | Petrowitsch_Casos OVNI Chilenos_2024-01-29.xlsx + Petrowitsch_Casos Latino_americanos_2024-01-29.xlsx
FORMAT_PETROWITSCH_LATAM = {
    "DIA":                  "integer — day of observation; null if unknown",
    "MES":                  "integer — month of observation; null if unknown",
    "ANNO":                 "integer — year of observation (AÑO column); null if unknown",
    "HORA":                 "string — local time (HH hs. format, sparse); null if unknown",
    "H":                    "integer — duration hours component; null if unknown",
    "M":                    "integer — duration minutes component; null if unknown",
    "S":                    "integer — duration seconds component; null if unknown",
    "LUGAR":                "string — city or locality name (free-text)",
    "REGION":               "string — Chilean administrative region (Roman numeral, e.g. X, RM); null if unknown",
    "COORDENADAS":          "string — approximate coordinates (free-text, e.g. 41º45,73º44); null if unknown",
    "J_V":                  "string — Vallée catalogue reference code (e.g. IB); null if unknown",
    "N_objetos":            "integer — number of objects observed; null if unknown",
    "Descripcion":          "string — free-text description of the phenomenon (Descripción column)",
    "Velocidad":            "string — speed descriptor (free-text, sparse); null if unknown",
    "DIRECC":               "string — direction of travel (free-text, sparse); null if unknown",
    "OTRAS_CARACTERISTICAS":"string — additional observed characteristics; null if unknown",
    "FOTO_FILM":            "string — photo/film indicators (M=Movie, F=photo codes); null if none",
    "TESTIGOS_N":           "integer — number of witnesses; null if unknown",
    "Ocupacion":            "string — witness occupation (sparse; Ocupación column); null if unknown",
    "POSIBLE_EXPLICACION":  "string — possible explanation or identification; null if unknown",
    "FUENTES":              "string — bibliographic source references (coded)",
}

# ── FORMAT_UFOCAT ──────────────────────────────────────────────────────────────
# UFOCAT 2023 — J. Allen Hynek Center for UFO Studies / Sun River Research Institute | ~146,000 rows | EN | ufocat2023.accdb + UFOCAT 2023.xlsx
FORMAT_UFOCAT = {
    # ── Record linkage ─────────────────────────────────────────────────────────
    "URN":       "integer — UFOCAT Record Number: permanent unique identifier for every record (primary key)",
    "PRN":       "integer — Primary Record Number: URN of the most authoritative record for this event; group records by PRN to deduplicate event counts",
    "X2":        "enum [0, 1, 2, 3] — primacy within an event block: 0=primary (use for counting), 1=independent secondary investigation, 2=references primary source, 3=references a secondary source",
    "IRN":       "integer — Indirect Record Number: URN most closely associated with the source cited by this record's author; null for primary entries",
    "X1":        "enum [=, -] — confidence of PRN–IRN match: ==clear match, -=possible match; null=primary entry with no indirect source",

    # ── Bibliographic provenance ───────────────────────────────────────────────
    "AUTHOR":    "string — surname (and first initial) of the investigator or editor responsible for this record; may be organisation/journal name if author unknown",
    "LEVEL":     "enum [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] — source type/accessibility: 0–2=confidential investigation files; 3=preferred open investigation file; 4=newspaper/magazine; 5=UFO periodical; 6=personal files; 7=first-edition book; 8=revised/paperback edition; 9=list-only no narrative; null=non-event record",
    "SOURCE":    "string — 12-character mnemonic code for the direct source (investigation file, periodical, or book); lookup in SOURCES auxiliary table",
    "PAGEVOL":   "string — position within the direct source (page number, serial number, or date-based code, depending on LEVEL)",
    "ISOURCE":   "string — 12-character mnemonic for the indirect (cited) source; special punctuation codes: *=firsthand written investigation, +=standardised witness form, apostrophe=scientific journal account, .=witness letter/interview, ,=newspaper account",
    "IPAGEVOL":  "string — position within the indirect source; same scheme as PAGEVOL",

    # ── Date / Time ────────────────────────────────────────────────────────────
    "YEAR":      "string — year of the event (4 chars): exact year e.g. 1975; trailing .=decade only; E=early, M=middle, L=late in decade",
    "MO":        "string — month (01–12); blank first col + season code: E=Jan–Apr, F=Oct–Dec, L=Sep–Dec, M=May–Aug, S=Apr–Jun, W=Jan–Mar",
    "DAY":       "string — day (01–31); blank first col + E=early, M=middle, L=late in month",
    "TIME":      "string — local event onset time (HHMM 24-hour); or descriptor: Day, Night, Morn, After, Even, Dawn, Noon, Dusk, MidD, MidN, Pdawn; - in minute sub-field=early, +=late",
    "TZ":        "enum [*, +, ., -, ', =] — date/time accuracy: *=GMT/UTC, +=daylight saving, .=confirmed local standard, -=date approximate, apostrophe=publication date, ==date believed erroneous; null=unspecified",
    "TZONE":     "float — timezone code (hours behind Greenwich noon): 4=Pacific Standard, 5=Mountain Standard, 6=Central Standard, 7=Eastern Standard; fractional values for non-integer UTC offsets",

    # ── Location ───────────────────────────────────────────────────────────────
    "LOCATION":  "string — place-name of the sighting (city/town, up to 30 chars); formatted PLACENAME DIRECTION DISTANCEunits (e.g. LOUISVILLE SW 10M); = separates two places meaning between",
    "LFLG":      "enum [-, +, *, =] — location accuracy: -=approximate, +=specific witness location, *=specific phenomenon location, ==place-name believed erroneous; null=standard",
    "REGION":    "enum [A, AA, AF, AS, AU, CA, CN, EU, I, M, ME, NA, P, SA, US, W, XX, RE] — world region: A=Atlantic, AA=Antarctica, AF=Africa, AS=Asia, AU=Australia+Oceania, CA=Central America, CN=Canada, EU=Europe, I=Indian Ocean, M=Mediterranean, ME=Middle East, NA=North America unspecified, P=Pacific, SA=South America, US=United States, W=World unspecified, XX=Outer Space, RE=Roman Empire",
    "STATE":     "string — state or country code (up to 3 chars): US states use 2-letter USPS abbreviations; countries use 1–3 letter codes from WORLD auxiliary table",
    "COUNTY":    "string — county or province (up to 12 chars); ~3,150 US counties; non-US uses province",
    "LONGITUDE": "string — longitude (up to 8 chars); default English system: West positive; decimal point in col 5=decimal degrees, apostrophe=degrees+minutes; see X3",
    "LATITUDE":  "string — latitude (up to 8 chars); same system as LONGITUDE; two-letter prefix=British National Grid",
    "X3":        "enum [M, Q, =] — coordinate system: null=English (360°, Greenwich), M=French metric (400°, Notre-Dame Paris), Q=English but questionable, ==known error",
    "TER":       "string — terrain/landmark type at sighting location (~100 values; e.g. Airfield, Airport, City, Electric, Farm, Forest, Highway, Lake, Military, Missile, Mountain, Nuclear, Ocean, River, Shore, Warplane, Warship, Woods); see codebook Table 5",

    # ── Witness ────────────────────────────────────────────────────────────────
    "WITNESSES": "string — number of witnesses (right-justified); -=about, +=at least; C=hundred, K=thousand; letter codes: F=few, G=group, M=many, N=numerous, S=several",
    "AGE":       "string — age of principal witness in years (2 chars); or group composition: first char=number of families, second=A/C/T/S/M/X/Y/W adult-type code",
    "SEX":       "enum [M, F, C, X] — witness sex/group: M=all male, F=all female, C=couple, X=mixed; null=unknown",
    "MIS":       "enum [#, +, =] — #=military witnesses, +=police witnesses, ==witness name incorrectly copied; null=none",
    "NAMES":     "string — principal witness surname(s); additional witnesses separated by =; confidential=initials only; organisation in parentheses; investigator name in reverse parentheses for confidential cases",
    "VFLG":      "enum [(, , ,', ., #] — vehicle flag: (=orbiting spacecraft, apostrophe=aircraft, ,=boat, .=automobile/motorcycle, #=airborne+ground witnesses in contact; null=none",

    # ── Phenomenology ──────────────────────────────────────────────────────────
    "HYNEK":     "enum [NL, ND, NO, DD, DL, DO, RV, RR, CE1, CE2, CE3, CE4, AM, BH, CM, CR, MIB, TC] — Hynek extended classification: NL=Nocturnal Light, ND=Nocturnal Disc, NO=Nocturnal Object, DD=Daylight Disc, DL=Daylight Light, DO=Daylight Object, RV=Radar-Visual, RR=Radar only, CE1–CE4=Close Encounters, AM=Animal Mutilation, BH=Black Helicopter, CM=Cattle Mutilation, CR=Crash Retrieval, MIB=Men in Black, TC=Physical Trace only",
    "VALLEE":    "enum [AN1, AN2, AN3, AN4, AN5, FB1, FB2, FB3, FB4, FB5, CE1, CE2, CE3, CE4, CE5, MA1, MA2, MA3, MA4, MA5] — Vallée classification: AN=Anomaly, FB=Flyby, CE=Close Encounter, MA=Manoeuvre; sub-type 1–5 indicates increasing physical/experiential involvement",
    "SVP":       "string — SVP Credibility Rating (3 digits): S=Source Reliability (0–4), V=Site Visit quality (0–4), P=Possible Explanations resisted (0–4); e.g. 342",
    "TYPE":      "string — UFOCAT type code: first digit 0–9=strangeness scale (0=non-UFO, 1=stationary, 2=moving, 3=single discontinuity, 4=multiple discontinuities, 5=encounter without landing, 6=landed, 7=occupants outside, 8=intelligent communication, 9=lasting physical effects); followed by letter sub-type codes (E=EM effect, T=physical trace, W=independent witnesses, etc.)",
    "EXPL":      "enum [=, -, +, *] — explainability flag: ==certainly explained, -=probably/possibly explained, null=no explanation offered, +=probably unconventional, *=virtually certain unconventional",
    "EXPLAN":    "enum [Aircraft, Animals, AnomProp, Aurora, Balloon, Birds, Blimp, Bolide, Cloud, Coincidenc, Comet, Debris, EM, FalseTarge, Fireball, Flare, Hallucinat, Helicopter, Hoax, Inversion, Junk, Jupiter, Lensflare, Light, Lights, Malfunct, Mars, Meteor, Mirage, Missile, Moon, Natural, Op-error, Planet, Prank, Psych, Quake, RadarChaff, Re-entry, Reflection, Rocket, Rumor, Satelite, Saturn, SpiderGoss, Star, Trawlers, Unreliable, Venus, Weather, Wind] — conventional explanation as given by the direct source; null if unexplained",
    "OBJS":      "string — number of UFO objects (right-justified; -=about, +=at least; C=hundred; letter codes F/G/M/N/S; 0=no object; for trace-only cases=number of trace marks)",
    "DUR":       "string — duration of observation (normally minutes; +=at least; decimal prefix=seconds; H suffix=hours; D=days; B=brief, VB=very brief, .F=few seconds, .S=several seconds, SH=several hours)",
    "SIZE":      "integer — estimated physical size as base-2 log of largest dimension in feet (e.g. 5=32 ft/10 m, 10=~1,000 ft/300 m); negative values possible; null if unknown",
    "ANGLSIZ":   "string — apparent (angular) size: .=point source, .0=non-point, .1=≈sun/moon, .2=twice sun/moon; Poher qualitative codes: (A=immense, (B=quarter-coin at arm's length, (C=orange, (D=melon/plate, (E=40 cm, (F=1 m, (G=2 m, (H=commercial aircraft",
    "DIST":      "integer — closest approach distance as base-2 log in feet (e.g. 8=256 ft/78 m; -1=witness touched UFO); null if unknown",
    "COLOR":     "enum [Aluminum, Amber, Beige, Black, Blue, BluGray, BluGreen, Bluish, Bright, Bronze, Brown, Changed, Chrome, Clear, Copper, Dark, Fire, Fluoresc, Glowing, Gold, Gray, Green, Luminous, Metallic, Multi, Orange, Pearl, Phosphor, Pink, Red, Rust, Shiny, Silver, Tan, Transpar, Violet, White, Yellow, Zinc] — principal colour; Multi=multi-coloured; Changed=colour changed during sighting; null if unknown",
    "SHAPE":     "enum [Aircraft, Arrow, Badge, Ball, Banana, Bat, Beam, Beehive, Blimp, Bottle, Bowl, BowTie, Box, Bulb, Bullet, Capsule, Cigar, Circle, Circular, Cloud, Cone, Copter, Crescent, Cross, Crown, Cube, Cylinder, Delta, Diamond, Dirigibl, Disc, DomeDisc, Dumbell, Ellipse, Elliptic, Elongate, FanBlade, Fish, Flare, Flash, Football, Fuselage, Globe, Hemisphr, L-Shape, Lampshad, Lens, Light, Lights, Linear, Lozenge, MantaRay, Missile, Object, Oblong, Odd, Orb, Oval, Ovoid, Pear, Polygon, Polymorf, Pyramid, Rectangl, Ring, Rocket, Round, Saturn, Schliere, Sparks, Sphere, Spindle, Square, Star, T-Shape, Teardrop, Top, Torpedo, Triangle, Tube, V-Shape, Walnut, Wedge, Wing, X-Shape, Y-Shape] — principal shape; Polymorf=shape-changing; V-Shape=boomerang; Polygon=all multi-sided; null if unknown",
    "SOUND":     "string — principal sound reported (e.g. NoSound, Noise, Humming, Buzzing, Whistle, Roar, Rumbling, Hissing, Jet, Droning, Whirring; NoSound=explicitly silent); null if unknown",
    "WEA":       "enum [Calm, Clear, Clouds, Cloudy, Fog, Ice, Muddy, Overcast, Partial, Rain, Snow, Storm, Thunder, Undrcast] — weather conditions; null if not recorded",

    # ── Entity fields (added 2002) ─────────────────────────────────────────────
    "Ufonaut":   "string — classification of UFO humanoid/entity type reported; null if no entity",
    "Height":    "float — reported height of the Ufonaut/entity in feet; null if none or unknown",
    "Uniform":   "string — uniform or clothing description of the Ufonaut/entity; null if none or unknown",

    # ── Computed temporal fields ────────────────────────────────────────────────
    "AST":         "float — Absolute Sidereal Time (4-digit integer rounded to nearest hour) calibrated to Puy-de-Dôme, France (Epsilon Eridani reference); null if uncomputable",
    "LST":         "float — Local Sidereal Time at the specific sighting location; null if uncomputable",
    "USER_DEFINED":"string — user-definable field; currently stores LST prefixed with RA for sidereal-time searches",

    # ── Free-text narrative ────────────────────────────────────────────────────
    "NOTES":     "string — free-text narrative describing key aspects of the report (up to 200 chars; added 1998; sparse); null if empty",
}

# ── FORMAT_CISU_CISUCAT ────────────────────────────────────────────────────────
# CISU Italian UAP catalogue (CISUCAT) | ~50,000+ rows estimated | IT | 2022-08-31 CISUCAT.xls + annual revisions
# Note: column schema requires XLS parsing; fields below are derived from CISU publications.
FORMAT_CISU_CISUCAT = {
    "case_number":      "string — CISU internal case number (exact column name TBC from XLS)",
    "date":             "string — date of observation",
    "time":             "string — local time of observation; null if unknown",
    "location":         "string — city or locality of sighting (Italy)",
    "province":         "string — Italian province abbreviation",
    "country":          "string — country (primarily Italy)",
    "shape":            "string — shape of the observed object (free-text); null if unknown",
    "duration":         "string — observed duration (free-text); null if unknown",
    "num_witnesses":    "integer — number of witnesses; null if unknown",
    "witness_type":     "string — type of witness (e.g. civilian, military, pilot); null if unknown",
    "hynek":            "string — Hynek classification code; null if unknown",
    "source":           "string — source of the report (bibliography or organisation)",
    "description":      "string — free-text narrative of the sighting (Italian)",
}

# ── FORMAT_CUFOC_1977_ITALY_FRANCE ────────────────────────────────────────────
# CUFOC 1977 Italy-France wave study (Bourdon+Delaval digitisation) | partial rows | FR/IT | ita_fr77_delaval.xlsx
# Note: column schema requires XLSX parsing; fields below are derived from Bourdon/Delaval methodology.
FORMAT_CUFOC_1977_ITALY_FRANCE = {
    "case_id":      "string — case identifier from the CUFOC/CUN coding schema",
    "date":         "string — date of observation (1977 wave)",
    "time":         "string — local time of observation; null if unknown",
    "country":      "enum [France, Italy] — country of the sighting",
    "location":     "string — city or locality (French or Italian)",
    "hynek":        "string — Hynek classification; null if unknown",
    "ifo_ufo":      "string — IFO/UFO determination (identified vs. unidentified); null if unknown",
    "shape":        "string — shape of observed object; null if unknown",
    "source":       "string — source publication or investigation reference (Bourdon/Delaval)",
    "description":  "string — free-text narrative of the sighting (French or Italian)",
}

# ── FORMAT_UAPCHECK ────────────────────────────────────────────────────────────
# UAPCHECK.com international data-exchange API registry | 22 organisations | EN | api.sql
# Note: this is an organisational registry, NOT a sighting catalogue.
FORMAT_UAPCHECK = {
    "id_api":          "integer — auto-increment organisation ID",
    "api_org_key":     "string — unique organisation key (format UAP-XXXXXX)",
    "api_org_name":    "string — human-readable organisation name (e.g. MUFON (USA), GEP (Germany), COBEPS (Belgium))",
    "api_org_url":     "string — website or API endpoint URL of the organisation",
    "api_org_lang":    "string — ISO 639-1 language code of the organisation",
    "api_org_latitude":"float — latitude of organisation headquarters",
    "api_org_longitude":"float — longitude of organisation headquarters",
    "api_options":     "string — JSON structure with API configuration options; null if none",
}
