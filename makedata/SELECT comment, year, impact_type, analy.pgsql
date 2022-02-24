SELECT comment, year, impact_type, analysis_id, main_sector, sub_sector, name, type, country_of_incorporation, current_analyse_id, entity_impact_id
FROM "impact_comments" as com 
    JOIN "entities" as ent ON com.entity_id = ent.id 
    WHERE com.comment IS NOT NULL  AND com.comment != "" AND ent.main_sector IS NOT NULL