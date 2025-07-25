use crate::{cache::get_cached_config, types::*};
use chrono::Utc;
use poe_api_process::{ModelInfo, get_model_list};
use salvo::prelude::*;
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

// Note: This cache doesn't apply to /api/models path
static API_MODELS_CACHE: RwLock<Option<Arc<Vec<ModelInfo>>>> = RwLock::const_new(None);

#[handler]
pub async fn get_models(req: &mut Request, res: &mut Response) {
    let path = req.uri().path();
    info!("📋 Received request to get model list | Path: {}", path);
    let start_time = Instant::now();

    // Handle /api/models special path (no cache) ---
    if path == "/api/models" {
        info!("⚡️ api/models path: directly from Poe (no cache)");
        match get_model_list(Some("en-us")).await {
            Ok(model_list) => {
                let lowercase_models = model_list
                    .data
                    .into_iter()
                    .map(|mut model| {
                        model.id = model.id.to_lowercase();
                        model
                    })
                    .collect::<Vec<_>>();

                let models_arc = Arc::new(lowercase_models);

                {
                    let mut cache_guard = API_MODELS_CACHE.write().await;
                    *cache_guard = Some(models_arc.clone());
                    info!("🔄 Updated API_MODELS_CACHE after /api/models request.");
                }

                let response = json!({
                    "object": "list",
                    "data": &*models_arc
                });

                let duration = start_time.elapsed();
                info!(
                    "✅ [/api/models] Successfully got unfiltered model list and updated cache | Model count: {} | Processing time: {}",
                    models_arc.len(), // Use Arc length
                    crate::utils::format_duration(duration)
                );
                res.render(Json(response));
            }
            Err(e) => {
                let duration = start_time.elapsed();
                error!(
                    "❌ [/api/models] Failed to get model list | Error: {} | Duration: {}",
                    e,
                    crate::utils::format_duration(duration)
                );
                res.status_code(StatusCode::INTERNAL_SERVER_ERROR);
                res.render(Json(json!({ "error": e.to_string() })));
            }
        }
        return;
    }

    let config = get_cached_config().await;

    let is_enabled = config.enable.unwrap_or(false);
    debug!("🔍 Config enabled status (from cache): {}", is_enabled);

    let yaml_config_map: std::collections::HashMap<String, ModelConfig> = config
        .models
        .clone() // Clone HashMap from Arc<Config>
        .into_iter()
        .map(|(k, v)| (k.to_lowercase(), v))
        .collect();

    if is_enabled {
        info!("⚙️ Merging cached Poe API list with models.yaml (enabled)");

        let api_models_data_arc: Arc<Vec<ModelInfo>>;

        let read_guard = API_MODELS_CACHE.read().await;
        if let Some(cached_data) = &*read_guard {
            // Cache hit
            debug!("✅ Model cache hit.");
            api_models_data_arc = cached_data.clone();
            drop(read_guard);
        } else {
            // Cache miss
            debug!("❌ Model cache miss. Attempting to populate...");
            drop(read_guard);

            let mut write_guard = API_MODELS_CACHE.write().await;
            // Check again to prevent race condition during write lock acquisition
            if let Some(cached_data) = &*write_guard {
                debug!("✅ API model cache filled by another thread during write lock wait.");
                api_models_data_arc = cached_data.clone();
            } else {
                // Cache is indeed empty, fetch from API
                info!("⏳ Fetching models from API to populate cache...");
                match get_model_list(Some("en-us")).await {
                    Ok(list) => {
                        let lowercase_models = list
                            .data
                            .into_iter()
                            .map(|mut model| {
                                model.id = model.id.to_lowercase();
                                model
                            })
                            .collect::<Vec<_>>();
                        let new_data = Arc::new(lowercase_models);
                        *write_guard = Some(new_data.clone());
                        api_models_data_arc = new_data;
                        info!("✅ API models cache populated successfully.");
                    }
                    Err(e) => {
                        // If cache population failed, return error
                        let duration = start_time.elapsed(); // Calculate duration
                        error!(
                            "❌ Failed to populate API models cache: {} | Duration: {}.",
                            e,
                            crate::utils::format_duration(duration) // Use duration in log
                        );
                        res.status_code(StatusCode::INTERNAL_SERVER_ERROR);
                        res.render(Json(
                            json!({ "error": format!("Failed to retrieve model list for cache population: {}", e) }),
                        ));
                        drop(write_guard);
                        return;
                    }
                }
            }
            drop(write_guard);
        }

        let mut api_model_ids: HashSet<String> = HashSet::new();
        for model_ref in api_models_data_arc.iter() {
            api_model_ids.insert(model_ref.id.to_lowercase());
        }

        let mut processed_models_enabled: Vec<ModelInfo> = Vec::new();

        for api_model_ref in api_models_data_arc.iter() {
            let api_model_id_lower = api_model_ref.id.to_lowercase();
            match yaml_config_map.get(&api_model_id_lower) {
                Some(yaml_config) => {
                    // Found in YAML: check if enabled, apply mapping if enabled
                    if yaml_config.enable.unwrap_or(true) {
                        let final_id = if let Some(mapping) = &yaml_config.mapping {
                            let new_id = mapping.to_lowercase();
                            debug!(
                                "🔄 API model renamed (YAML enabled): {} -> {}",
                                api_model_id_lower, new_id
                            );
                            new_id
                        } else {
                            debug!(
                                "✅ Keep API model (YAML enabled, no mapping): {}",
                                api_model_id_lower
                            );
                            api_model_id_lower.clone()
                        };
                        processed_models_enabled.push(ModelInfo {
                            id: final_id,
                            object: api_model_ref.object.clone(),
                            created: api_model_ref.created,
                            owned_by: api_model_ref.owned_by.clone(),
                        });
                    } else {
                        debug!("❌ Exclude API model (YAML disabled): {}", api_model_id_lower);
                    }
                }
                None => {
                    debug!("✅ Keep API model (not in YAML): {}", api_model_id_lower);
                    processed_models_enabled.push(ModelInfo {
                        id: api_model_id_lower.clone(),
                        object: api_model_ref.object.clone(),
                        created: api_model_ref.created,
                        owned_by: api_model_ref.owned_by.clone(),
                    });
                }
            }
        }

        // Process custom models, add to processed models list
        if let Some(custom_models) = &config.custom_models {
            if !custom_models.is_empty() {
                info!("📋 Processing custom models | Count: {}", custom_models.len());
                for custom_model in custom_models {
                    let model_id = custom_model.id.to_lowercase();
                    // Check if this ID already exists in processed models
                    if !processed_models_enabled.iter().any(|m| m.id == model_id) {
                        // Check if enable: false configured in yaml_config_map
                        if let Some(yaml_config) = yaml_config_map.get(&model_id) {
                            if yaml_config.enable == Some(false) {
                                debug!("❌ Exclude custom model (YAML disabled): {}", model_id);
                                continue;
                            }
                        }

                        debug!("➕ Adding custom model: {}", model_id);
                        processed_models_enabled.push(ModelInfo {
                            id: model_id,
                            object: "model".to_string(),
                            created: custom_model
                                .created
                                .unwrap_or_else(|| Utc::now().timestamp()),
                            owned_by: custom_model
                                .owned_by
                                .clone()
                                .unwrap_or_else(|| "poe".to_string()),
                        });
                    }
                }
            }
        }

        let response = json!({
            "object": "list",
            "data": processed_models_enabled
        });

        let duration = start_time.elapsed();
        info!(
            "✅ Successfully got processed model list | Source: {} | Model count: {} | Processing time: {}",
            "YAML + Cached API",
            processed_models_enabled.len(),
            crate::utils::format_duration(duration)
        );

        res.render(Json(response));
    } else {
        info!("🔌 YAML disabled, directly getting model list from Poe API (no cache, no YAML rules)...");

        match get_model_list(Some("en-us")).await {
            Ok(model_list) => {
                let lowercase_models = model_list
                    .data
                    .into_iter()
                    .map(|mut model| {
                        model.id = model.id.to_lowercase();
                        model
                    })
                    .collect::<Vec<_>>();

                let response = json!({
                    "object": "list",
                    "data": lowercase_models
                });
                let duration = start_time.elapsed();
                info!(
                    "✅ [Direct Poe] Successfully got model list directly | Model count: {} | Processing time: {}",
                    lowercase_models.len(),
                    crate::utils::format_duration(duration)
                );
                res.render(Json(response));
            }
            Err(e) => {
                let duration = start_time.elapsed();
                error!(
                    "❌ [Direct Poe] Failed to get model list directly | Error: {} | Duration: {}",
                    e,
                    crate::utils::format_duration(duration)
                );
                res.status_code(StatusCode::INTERNAL_SERVER_ERROR);
                res.render(Json(
                    json!({ "error": format!("Failed to get models directly from API: {}", e) }),
                ));
            }
        }
    }
}
