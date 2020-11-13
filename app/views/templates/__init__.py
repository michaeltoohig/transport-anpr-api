from starlette.templating import Jinja2Templates

from app.views.templates.filters import css_script, js_script

templates = Jinja2Templates(directory="app/views/templates")

# Add custom filters to Jinja templates
templates.env.filters["css_script"] = css_script
templates.env.filters["js_script"] = js_script
