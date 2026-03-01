from simulator.gpt_structure import *
import pickle
import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

def load_locations(csv_filename):
    valid_locations = set()
    with open(csv_filename, 'r', encoding='utf-8') as f:
        for line in f:
            loc = line.strip()
            if loc:
                valid_locations.add(loc)
    return valid_locations

valid_locations = load_locations("subcategories.csv")
root_directory = "./simulator/"

def valid_generation(data):
    """Validate plan items for time format and allowed locations"""
    pattern = re.compile(r'(?P<location_full>[^#]+#\d+)\s+at\s+(?P<time>([0-1]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)')
    for item in data[0]:
        lower = item.lower()
    for item in data:
        match = pattern.match(item)
        if match is None:
            return False
        location_full = match.group("location_full")
        if '#' not in location_full:
            return False
        elif re.search(r'\b(?:AM|PM)\b', item, re.IGNORECASE):
            return False
        location_name = location_full.split('#')[0].strip()
        if location_name not in valid_locations:
            return False
    return True

def check_workday_or_weekend(date_str: str) -> str:
    """Return Weekday or Weekend."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() < 5:
        return "Weekday"
    else:
        return "Weekend"

def get_long_routines(date_, test_routine_list, num_days=2):
    current_date = datetime.strptime(date_, "%Y-%m-%d")
    routines_with_diff = []
    for test_route in test_routine_list:
        date_str = test_route.split(": ")[0].split(" ")[-1]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj == current_date:
            continue
        days_diff = (current_date - date_obj).days
        if days_diff > 0:
            routines_with_diff.append((test_route, days_diff))
    routines_with_diff.sort(key=lambda x: x[1])
    output_routines = [route[0] for route in routines_with_diff[num_days:]]
    return output_routines

def get_recent_routines(date_, test_routine_list, num_days=2):
    current_date = datetime.strptime(date_, "%Y-%m-%d")
    routines_with_diff = []
    for test_route in test_routine_list:
        date_str = test_route.split(": ")[0].split(" ")[-1]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj == current_date:
            continue
        days_diff = (current_date - date_obj).days
        if days_diff > 0:
            routines_with_diff.append((test_route, days_diff))
    routines_with_diff.sort(key=lambda x: x[1])
    output_routines = [route[0] for route in routines_with_diff[:num_days]]
    if len(output_routines) < num_days:
        output_routines = [route[0] for route in routines_with_diff]
    return output_routines

class DayPlannerConfig:
    """Initialize the planner with a configuration."""
    MAX_TRIAL = 3
    MAX_REFLECTION_TRY = 3
    REPLAN_TRIAL = 3

    REPLAN_TEMPLATE = "./simulator/prompt_template/regeneration.txt"
    GENERATION_TEMPLATE = "./simulator/prompt_template/generation.txt"
    EVENT_SCHEMA_TEMPLATE  = "./simulator/prompt_template/event_schema.txt"
    EVENT_GIST_TEMPLATE = "./simulator/prompt_template/event_gist.txt"
    PATTERN_GIST_TEMPLATE = "./simulator/prompt_template/pattern_gist.txt"
    ACTION_GIST_TEMPLATE = "./simulator/prompt_template/action_gist.txt"
    REFLECTION_TEMPLATE = "./simulator/prompt_template/reflection_alignment.txt"

class DayPlanner:
    """Plan daily activities."""
    def __init__(self, config: DayPlannerConfig = None):
        self.config = config or DayPlannerConfig()

    def plan_new_day(self, person, sample_num: int = 1) -> Dict[str, Dict[str, str]]:
        world_interaction = self._initialize_world_interaction()
        for k in range(sample_num):
            for test_route in person.test_routine_list:
                date = self._extract_date(test_route)
                recent_routine = get_recent_routines(date, person.train_routine_list)
                long_routine = get_long_routines(date, person.train_routine_list)
                event_summary, event_gist, day_type = self._get_event_summary(date)
                pattern_data_input = [person.train_routine_list, recent_routine]
                prompt_pattern_gist = generate_prompt(pattern_data_input, self.config.PATTERN_GIST_TEMPLATE)
                pattern_gist_contents = execute_prompt(prompt_pattern_gist, objective=f"INFER_PATTERN_GIST")

                try:
                    plan_result, reason = self._generate_initial_plan(
                        recent_routine, long_routine, event_summary, day_type
                    )
                except:
                    self._use_fallback_plan(person, date, test_route, world_interaction)
                    continue
                validated_plan = self._validate_and_replan(
                    plan_result, event_summary, person, recent_routine, long_routine, day_type, reason, event_gist, pattern_gist_contents
                )

                if validated_plan is None:
                    self._use_fallback_plan(person, date, test_route, world_interaction)
                else:
                    self._save_successful_plan(validated_plan, date, test_route, world_interaction)
                self._update_training_data(person, test_route)
        return world_interaction

    def _initialize_world_interaction(self) -> Dict[str, Dict[str, str]]:
        return {"results": {}, "reals": {}}

    def _extract_date(self, test_route: str) -> str:
        return test_route.split(": ")[0].split(" ")[-1]

    def _get_recent_routine(self, date: str, train_routine_list: List[str]) -> str:
        recent_routine = get_recent_routines(date, train_routine_list)
        return parse_activities(recent_routine)


    def _get_event_summary(self, date) -> str:
        """Build event schema and event gist with prompts and label the day type."""
        event_context = "Put event context here."
        curr_input = [event_context]
        prompt_event_schema = generate_prompt(curr_input, self.config.EVENT_SCHEMA_TEMPLATE)
        event_schema_contents = execute_prompt(prompt_event_schema, objective=f"INFER_EVENT_SCHEMA")
        prompt_event_gist = generate_prompt(curr_input, self.config.EVENT_GIST_TEMPLATE)
        event_gist_contents = execute_prompt(prompt_event_gist, objective=f"INFER_EVENT_GIST")
        day_type = f"Today is {check_workday_or_weekend(date)}."
        return event_schema_contents, event_gist_contents, day_type

    def _generate_initial_plan(
            self, recent_routine: str, history_routine: str,
            event_summary: str, day_type: str
    ) -> Optional[List[str]]:
        """Try several generations and return a plan with its reason."""
        curr_input = [history_routine, recent_routine, event_summary, day_type]
        prompt = generate_prompt(curr_input, self.config.GENERATION_TEMPLATE)

        for trial in range(self.config.MAX_TRIAL):
            contents = execute_prompt(prompt, objective=f"INFER_RE_{trial}")
            print(contents)
            if not contents:
                print("No content found. Regenerating prompt and trying again...")
                continue
            try:
                parsed_data = json.loads(contents)
                plan = parsed_data["plan"]
                reason = parsed_data["reason"]
                if valid_generation(plan):
                    return plan, reason
                else:
                    print(contents)
                    print("Invalid format and trying again initial generation...")

            except json.JSONDecodeError:
                print("Invalid JSON format and trying again initial generation...")
                print(contents)
                continue

        return None


    def _validate_and_replan(
            self, initial_plan: List[str], event_summary: str, person,
            recent_routine: str, history_routine: str, day_type:str, reason:str, event_gist, pattern_gist_contents
    ) -> Optional[List[str]]:
        """Run reflection then replan if needed."""
        current_plan = initial_plan
        for attempt in range(self.config.MAX_REFLECTION_TRY):
            reflection_result = self._run_reflection_validation(current_plan, event_gist, pattern_gist_contents, reason)
            if reflection_result and self._is_reflection_successful(reflection_result):
                return current_plan
            reason = reflection_result.get("reason") if reflection_result else "Reflection failed"
            try:
                result = self._replan_activities(
                    recent_routine, history_routine, event_summary,
                    current_plan, reason, day_type
                )
                if result is None:
                    break
                replanned, reason = result
            except Exception:
                break

            current_plan = replanned

        return None

    def _run_reflection_validation(self, plan, event_gist_content, pattern_gist_contents, reason):
        """Execute reflection prompts on the plan and parse the returned JSON into a dict."""
        try:
            curr_action_input = [plan, reason]
            prompt_action_gist = generate_prompt(curr_action_input, self.config.ACTION_GIST_TEMPLATE)
            action_gist_contents = execute_prompt(prompt_action_gist, objective=f"INFER_ACTION_GIST")
            reflection_inputs = [event_gist_content, pattern_gist_contents, action_gist_contents]
            reflection_prompt = generate_prompt(reflection_inputs, self.config.REFLECTION_TEMPLATE)
            reflection_raw = execute_prompt(reflection_prompt, objective="REFLECTION")
            s = reflection_raw.strip()
            s = re.sub(r"```json", "", s)
            s = s.replace("```", "")
            m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
            if not m:
                clean = s
            else:
                clean = m.group(1)
            return json.loads(clean)
        except json.JSONDecodeError:
            return None

    def _is_reflection_successful(self, reflection_result: Dict[str, Any]) -> bool:
        required_keys = {"coherence_with_pattern", "coherence_with_event", "reason"}
        if not required_keys.issubset(reflection_result.keys()):
            return False

        coherence_judgment = reflection_result.get("coherence_with_pattern")
        alignment_judgment = reflection_result.get("coherence_with_event")

        if not isinstance(coherence_judgment, bool) or not isinstance(alignment_judgment, bool):
            return False

        return coherence_judgment and alignment_judgment

    def _replan_activities(
            self, recent_routine: str, history_routine: str, event_summary: str,
            current_plan: List[str], reason: str, day_type: str
    ) -> Optional[List[str]]:
        gen_inputs = [history_routine, recent_routine, event_summary, day_type,
                      current_plan, reason]
        replan_prompt = generate_prompt(gen_inputs, self.config.REPLAN_TEMPLATE)


        for trial in range(self.config.REPLAN_TRIAL):
            replan_raw = execute_prompt(replan_prompt, objective="REGENERATION")
            if not replan_raw:
                continue

            try:
                parsed_data = json.loads(replan_raw)
                plan = parsed_data["plan"]
                reason = parsed_data["reason"]
                if valid_generation(plan):
                    return plan, reason
                else:
                    print(replan_raw)
                    print("Invalid format and trying again...")

            except json.JSONDecodeError:
                print("Invalid JSON format and trying again...")
                print(replan_raw)
                continue

        return None

    def _use_fallback_plan(
            self, person, date: str, test_route: str,
            world_interaction: Dict[str, Dict[str, str]]
    ) -> None:
        try:
            old = person.train_routine_list[-1]
            date_token = old.split()[2]
            new = old.replace(date_token, f"{date}:", 1)

            if new.count('#') == 1:
                header, rest = new.split(': ', 1)
                rest = rest.rstrip('.')
                new = f"{header}: {rest}, {rest}."
        except:
            new = f"Activities at {date}: "

        world_interaction["results"][date] = new
        world_interaction["reals"][date] = test_route

        print("---------USE OLD------------")
        print(new)
        print(test_route)

    def _save_successful_plan(
            self, plan: List[str], date: str, test_route: str,
            world_interaction: Dict[str, Dict[str, str]]
    ) -> None:
        print("Plan:", plan)
        print("True:", test_route)
        print("Date:", date)
        world_interaction["reals"][date] = test_route
        world_interaction["results"][date] = f"Activities at {date}: " + ', '.join(plan)
        print("----------------------------------")

    def _update_training_data(self, person, test_route: str) -> None:
        """Append the past route to training data ."""
        person.train_routine_list.append(test_route)





