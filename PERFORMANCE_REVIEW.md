# SIPERI Performance Review and Optimization Notes

## Performance issues found

1. Server-side charts were regenerated on every GET and POST request.
   - `membership_plot()` and `feature_importance_plot()` ran even when the same chart already existed.
   - `result_plot()` overwrote one shared `result_chart.png` file, which is unsafe if multiple users evaluate at the same time.

2. Templates forced unnecessary image reloads.
   - Image URLs used `?v={{ range(1,999999)|random }}`.
   - This disables browser caching and makes every page load fetch the same images again.

3. Shared output filenames reduced scalability.
   - `membership.png`, `feature_importance.png`, and `result_chart.png` were overwritten globally.
   - Concurrent users or rapid requests could see another user's chart.

4. The main logo was oversized for the rendered UI.
   - Original `siperi-logo-main.png` was about 1 MB at 1254 x 1254 px.
   - It was resized to 360 x 360 px and optimized, reducing static payload substantially.

5. The Random Forest model used more trees than necessary for the current small sample data.
   - `n_estimators` was reduced from 120 to 80.
   - `n_jobs=-1` was added to use available CPU cores during model training.

6. The fuzzy inference object was recreated on every main evaluation.
   - The optimized version stores one `ActivityFuzzySystem` instance per activity in `AI_CACHE`.

## Optimizations applied

1. Added content-addressed chart caching.
   - Membership charts now use filenames based on activity, selected feature, and membership parameter hash.
   - Feature-importance charts now use filenames based on activity and model importance hash.
   - Result charts use a hash of the evaluated output values.

2. Removed browser cache-busting random query strings from templates.

3. Reduced chart image size.
   - Matplotlib figures now use smaller dimensions and `dpi=120` instead of `dpi=140`.

4. Optimized Random Forest training.
   - Changed to `RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)`.

5. Reused cached fuzzy system objects.

6. Removed duplicate `numpy` import.

## Expected impact

- Faster page loads after first chart generation.
- Lower CPU usage because repeated requests reuse chart files.
- Lower browser/network load because static images can now be cached.
- Safer behavior under multiple users because chart filenames are no longer overwritten globally.
- Smaller application package due to optimized logo.

## Remaining recommended improvements

1. Move charts to client-side rendering with Chart.js or SVG for near-zero server rendering overhead.
2. Add a cleanup policy for hashed result chart files if the app will be used by many users.
3. Train ML models offline and load saved `.joblib` files at startup if real datasets become large.
4. Split `app.py` into modules: config, fuzzy logic, ML, RL, routes, and plotting.
5. Use production deployment with Gunicorn instead of Flask development server for multi-user use.
