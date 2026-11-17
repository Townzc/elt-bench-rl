"""Task specifications of the credential-free local benchmark.

Each task mirrors the shape of an ELT-Bench task at a small scale: source
tables spread over heterogeneous systems, and data models that need joins,
aggregation, NULL handling and deterministic tie-breaking.

``sql`` on a data model is the reference transformation in DuckDB SQL, with
``{s}`` standing for the schema holding the loaded source tables. It is used
at build time only, to compute ground truth; it is never shown to the agent.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

SourceKind = Literal["sqlite", "documents", "object_store", "file"]
Rows = list[dict[str, Any]]


@dataclass(frozen=True)
class Column:
    name: str
    description: str


@dataclass(frozen=True)
class SourceTable:
    name: str
    kind: SourceKind
    columns: tuple[Column, ...]
    fmt: Literal["csv", "jsonl"] = "csv"
    """File format, for ``file`` sources."""


@dataclass(frozen=True)
class DataModel:
    name: str
    description: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    sql: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    tables: tuple[SourceTable, ...]
    models: tuple[DataModel, ...]
    generate: Callable[[random.Random], dict[str, Rows]]


def _date(rng: random.Random, year: int = 2024) -> str:
    return f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


_FIRST = ["Ada", "Ben", "Chloe", "Dev", "Elif", "Finn", "Grace", "Hugo", "Iris", "Jon",
          "Kira", "Liam", "Mona", "Nico", "Omar", "Pia", "Quinn", "Rosa", "Sam", "Tara"]  # fmt: skip
_LAST = ["Abbot", "Brooks", "Chen", "Diaz", "Evans", "Fox", "Garcia", "Hale", "Ito", "Jones",
         "Khan", "Lopez", "Meyer", "Novak", "Ortiz", "Park", "Reyes", "Silva", "Tran", "Wolfe"]  # fmt: skip


# ---------------------------------------------------------------------------------
# bookshop
# ---------------------------------------------------------------------------------
def _gen_bookshop(rng: random.Random) -> dict[str, Rows]:
    methods = [
        {"method_id": 1, "method_name": "Standard", "cost": 5.9},
        {"method_id": 2, "method_name": "Priority", "cost": 8.9},
        {"method_id": 3, "method_name": "Express", "cost": 11.9},
        {"method_id": 4, "method_name": "International", "cost": 24.5},
    ]
    publishers = [
        {"publisher_id": i + 1, "publisher_name": f"{name} Press"}
        for i, name in enumerate(
            ["Alder", "Birch", "Cedar", "Dogwood", "Elm", "Fir", "Ginkgo", "Hazel", "Ivy",
             "Juniper", "Kapok", "Larch"]  # fmt: skip
        )
    ]
    words = ["Silent", "Hidden", "Golden", "Broken", "Distant", "Last", "First", "Winter",
             "Summer", "Paper", "Glass", "Iron", "Salt", "River", "Night", "Morning"]  # fmt: skip
    nouns = ["Garden", "Harbor", "Letters", "Maps", "Orchard", "Kingdom", "Voyage", "Archive",
             "Lantern", "Mirror", "Station", "Island"]  # fmt: skip
    books = []
    for book_id in range(1, 81):
        books.append(
            {
                "book_id": book_id,
                "title": f"The {rng.choice(words)} {rng.choice(nouns)} {book_id}",
                # The last publisher has no books (exercises NULL/0 handling).
                "publisher_id": rng.randint(1, len(publishers) - 1),
                "num_pages": rng.choice([120, 180, 240, 240, 320, 320, 410, 560]),
                "publication_date": _date(rng, rng.randint(1990, 2023)),
                "language": rng.choice(["English", "English", "English", "Spanish", "Japanese"]),
            }
        )
    customers = []
    for customer_id in range(1, 61):
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        customers.append(
            {
                "customer_id": customer_id,
                "first_name": first,
                "last_name": last,
                "email": f"{first}.{last}{customer_id}@example.com".lower(),
            }
        )
    orders, lines = [], []
    line_id = 1
    for order_id in range(1, 151):
        orders.append(
            {
                "order_id": order_id,
                # Customers 51-60 never order.
                "customer_id": rng.randint(1, 50),
                "order_date": _date(rng),
                "shipping_method_id": rng.randint(1, 4),
            }
        )
        for _ in range(rng.randint(1, 4)):
            lines.append(
                {
                    "line_id": line_id,
                    "order_id": order_id,
                    "book_id": rng.randint(1, 80),
                    "price": round(rng.uniform(4, 40), 2),
                }
            )
            line_id += 1
    return {
        "customer": customers,
        "cust_order": orders,
        "order_line": lines,
        "book": books,
        "publisher": publishers,
        "shipping_method": methods,
    }


BOOKSHOP = TaskSpec(
    name="bookshop",
    tables=(
        SourceTable("customer", "sqlite", (
            Column("customer_id", "the unique identifier of the customer"),
            Column("first_name", "the first name of the customer"),
            Column("last_name", "the last name of the customer"),
            Column("email", "the email of the customer"),
        )),
        SourceTable("cust_order", "sqlite", (
            Column("order_id", "the unique identifier of the customer order"),
            Column("customer_id", "the id of the customer who placed the order"),
            Column("order_date", "the date of the order, YYYY-MM-DD"),
            Column("shipping_method_id", "the id of the shipping method used"),
        )),
        SourceTable("order_line", "sqlite", (
            Column("line_id", "the unique identifier of the order line"),
            Column("order_id", "the id of the order the line belongs to"),
            Column("book_id", "the id of the book ordered"),
            Column("price", "the price paid for the book on this line"),
        )),
        SourceTable("book", "documents", (
            Column("book_id", "the unique identifier of the book"),
            Column("title", "the title of the book"),
            Column("publisher_id", "the id of the publisher"),
            Column("num_pages", "the number of pages"),
            Column("publication_date", "the publication date, YYYY-MM-DD"),
            Column("language", "the language the book is written in"),
        )),
        SourceTable("publisher", "file", (
            Column("publisher_id", "the unique identifier of the publisher"),
            Column("publisher_name", "the name of the publisher"),
        ), fmt="csv"),
        SourceTable("shipping_method", "object_store", (
            Column("method_id", "the unique identifier of the shipping method"),
            Column("method_name", "the name of the shipping method"),
            Column("cost", "the cost of the shipping method"),
        )),
    ),  # fmt: skip
    models=(
        DataModel(
            name="customers",
            description="Each record represents a customer.",
            key=("customer_id",),
            columns=(
                Column("customer_id", "Unique identifier for the customer."),
                Column("customer_name", "Name of the customer. Concatenate the first name and last name with a space in between."),
                Column("email", "Email of the customer."),
                Column("num_orders", "Number of orders placed by the customer, replace NULL values with 0."),
                Column("total_spent", "Total price of all order lines in the customer's orders, rounded to 2 decimal places, replace NULL values with 0."),
                Column("preferred_shipping_method", "Name of the shipping method the customer used for the most orders, with ties broken by the ascending order of the shipping method name. NULL if the customer has no orders."),
            ),
            sql="""
with orders as (
    select customer_id, count(*) as num_orders from {s}.cust_order group by customer_id
), spend as (
    select o.customer_id, sum(l.price) as total_spent
    from {s}.cust_order o join {s}.order_line l on l.order_id = o.order_id
    group by o.customer_id
), ship as (
    select customer_id, method_name from (
        select o.customer_id, m.method_name,
               row_number() over (
                   partition by o.customer_id order by count(*) desc, m.method_name asc
               ) as rn
        from {s}.cust_order o
        join {s}.shipping_method m on m.method_id = o.shipping_method_id
        group by o.customer_id, m.method_name
    ) ranked where rn = 1
)
select c.customer_id,
       c.first_name || ' ' || c.last_name as customer_name,
       c.email,
       coalesce(orders.num_orders, 0) as num_orders,
       round(coalesce(spend.total_spent, 0), 2) as total_spent,
       ship.method_name as preferred_shipping_method
from {s}.customer c
left join orders on orders.customer_id = c.customer_id
left join spend on spend.customer_id = c.customer_id
left join ship on ship.customer_id = c.customer_id
""",
        ),
        DataModel(
            name="publishers",
            description="Each record represents a publisher.",
            key=("publisher_id",),
            columns=(
                Column("publisher_id", "Unique identifier for the publisher."),
                Column("publisher_name", "Name of the publisher."),
                Column("num_books", "Number of books published by the publisher, replace NULL values with 0."),
                Column("avg_num_pages", "Average number of pages of the publisher's books, rounded to 4 decimal places. NULL if the publisher has no books."),
                Column("longest_book_title", "Title of the publisher's book with the most pages, with ties broken by the ascending order of the title. NULL if the publisher has no books."),
                Column("num_copies_sold", "Number of order lines for the publisher's books, replace NULL values with 0."),
                Column("pct_english_books", "Percentage (0-100) of the publisher's books written in English, rounded to 4 decimal places. NULL if the publisher has no books."),
            ),
            sql="""
with stats as (
    select publisher_id,
           count(*) as num_books,
           avg(num_pages) as avg_num_pages,
           100.0 * sum(case when language = 'English' then 1 else 0 end) / count(*) as pct_english
    from {s}.book group by publisher_id
), longest as (
    select publisher_id, title from (
        select publisher_id, title,
               row_number() over (partition by publisher_id order by num_pages desc, title asc) as rn
        from {s}.book
    ) ranked where rn = 1
), sold as (
    select b.publisher_id, count(*) as num_copies_sold
    from {s}.order_line l join {s}.book b on b.book_id = l.book_id
    group by b.publisher_id
)
select p.publisher_id,
       p.publisher_name,
       coalesce(stats.num_books, 0) as num_books,
       round(stats.avg_num_pages, 4) as avg_num_pages,
       longest.title as longest_book_title,
       coalesce(sold.num_copies_sold, 0) as num_copies_sold,
       round(stats.pct_english, 4) as pct_english_books
from {s}.publisher p
left join stats on stats.publisher_id = p.publisher_id
left join longest on longest.publisher_id = p.publisher_id
left join sold on sold.publisher_id = p.publisher_id
""",
        ),
    ),  # fmt: skip
    generate=_gen_bookshop,
)


# ---------------------------------------------------------------------------------
# clinic
# ---------------------------------------------------------------------------------
def _gen_clinic(rng: random.Random) -> dict[str, Rows]:
    specialties = [
        {"specialty_id": i + 1, "specialty_name": name}
        for i, name in enumerate(["Cardiology", "Dermatology", "Neurology", "Pediatrics", "Oncology"])
    ]
    doctors = [
        {
            "doctor_id": 100 + i,
            "doctor_name": f"Dr. {rng.choice(_FIRST)} {rng.choice(_LAST)}",
            "specialty_id": rng.randint(1, 5),
            "years_experience": rng.randint(1, 35),
        }
        for i in range(14)
    ]
    cities = ["Albany", "Boulder", "Camden", "Dayton", "Eugene"]
    patients = [
        {
            "patient_id": i + 1,
            "full_name": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
            "birth_date": _date(rng, rng.randint(1945, 2015)),
            "city": rng.choice(cities),
        }
        for i in range(70)
    ]
    drugs = ["Amoxil", "Brufen", "Cetiriz", "Doxyl", "Enalap", "Fluox"]
    visits, prescriptions = [], []
    prescription_id = 1
    for visit_id in range(1, 261):
        visits.append(
            {
                "visit_id": visit_id,
                # Patients 61-70 never visit; the last two doctors see nobody.
                "patient_id": rng.randint(1, 60),
                "doctor_id": 100 + rng.randint(0, 11),
                "visit_date": _date(rng),
                "fee": round(rng.choice([45, 60, 80, 120, 150]) * rng.uniform(0.9, 1.1), 2),
            }
        )
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            prescriptions.append(
                {
                    "prescription_id": prescription_id,
                    "visit_id": visit_id,
                    "drug_name": rng.choice(drugs),
                    "quantity": rng.randint(1, 6) * 10,
                }
            )
            prescription_id += 1
    return {
        "patient": patients,
        "visit": visits,
        "doctor": doctors,
        "specialty": specialties,
        "prescription": prescriptions,
    }


CLINIC = TaskSpec(
    name="clinic",
    tables=(
        SourceTable("patient", "sqlite", (
            Column("patient_id", "the unique identifier of the patient"),
            Column("full_name", "the full name of the patient"),
            Column("birth_date", "the birth date of the patient, YYYY-MM-DD"),
            Column("city", "the city the patient lives in"),
        )),
        SourceTable("visit", "sqlite", (
            Column("visit_id", "the unique identifier of the visit"),
            Column("patient_id", "the id of the patient"),
            Column("doctor_id", "the id of the doctor seen"),
            Column("visit_date", "the date of the visit, YYYY-MM-DD"),
            Column("fee", "the fee charged for the visit"),
        )),
        SourceTable("doctor", "documents", (
            Column("doctor_id", "the unique identifier of the doctor"),
            Column("doctor_name", "the name of the doctor"),
            Column("specialty_id", "the id of the doctor's specialty"),
            Column("years_experience", "years of professional experience"),
        )),
        SourceTable("specialty", "file", (
            Column("specialty_id", "the unique identifier of the specialty"),
            Column("specialty_name", "the name of the specialty"),
        ), fmt="jsonl"),
        SourceTable("prescription", "object_store", (
            Column("prescription_id", "the unique identifier of the prescription"),
            Column("visit_id", "the id of the visit the prescription was written in"),
            Column("drug_name", "the name of the prescribed drug"),
            Column("quantity", "the number of units prescribed"),
        )),
    ),  # fmt: skip
    models=(
        DataModel(
            name="doctors",
            description="Each record represents a doctor.",
            key=("doctor_id",),
            columns=(
                Column("doctor_id", "Unique identifier for the doctor."),
                Column("doctor_name", "Name of the doctor."),
                Column("specialty_name", "Name of the doctor's specialty."),
                Column("num_visits", "Number of visits handled by the doctor, replace NULL values with 0."),
                Column("num_patients", "Number of distinct patients seen by the doctor, replace NULL values with 0."),
                Column("total_fees", "Sum of the fees of the doctor's visits, rounded to 2 decimal places, replace NULL values with 0."),
                Column("busiest_month", "Month (format YYYY-MM) in which the doctor had the most visits, with ties broken by the earliest month. NULL if the doctor has no visits."),
            ),
            sql="""
with stats as (
    select doctor_id, count(*) as num_visits, count(distinct patient_id) as num_patients,
           sum(fee) as total_fees
    from {s}.visit group by doctor_id
), months as (
    select doctor_id, month from (
        select doctor_id, strftime(cast(visit_date as date), '%Y-%m') as month,
               row_number() over (
                   partition by doctor_id
                   order by count(*) desc, strftime(cast(visit_date as date), '%Y-%m') asc
               ) as rn
        from {s}.visit
        group by doctor_id, strftime(cast(visit_date as date), '%Y-%m')
    ) ranked where rn = 1
)
select d.doctor_id,
       d.doctor_name,
       s.specialty_name,
       coalesce(stats.num_visits, 0) as num_visits,
       coalesce(stats.num_patients, 0) as num_patients,
       round(coalesce(stats.total_fees, 0), 2) as total_fees,
       months.month as busiest_month
from {s}.doctor d
left join {s}.specialty s on s.specialty_id = d.specialty_id
left join stats on stats.doctor_id = d.doctor_id
left join months on months.doctor_id = d.doctor_id
""",
        ),
        DataModel(
            name="patients",
            description="Each record represents a patient.",
            key=("patient_id",),
            columns=(
                Column("patient_id", "Unique identifier for the patient."),
                Column("full_name", "Full name of the patient."),
                Column("num_visits", "Number of visits of the patient, replace NULL values with 0."),
                Column("first_visit_date", "Date of the patient's earliest visit (format YYYY-MM-DD). NULL if the patient has no visits."),
                Column("num_doctors_seen", "Number of distinct doctors the patient has seen, replace NULL values with 0."),
                Column("total_drug_quantity", "Total quantity of all drugs prescribed in the patient's visits, replace NULL values with 0."),
                Column("most_prescribed_drug", "Name of the drug with the highest total prescribed quantity for the patient, with ties broken by the ascending order of the drug name. NULL if the patient has no prescriptions."),
            ),
            sql="""
with stats as (
    select patient_id, count(*) as num_visits,
           min(cast(visit_date as date)) as first_visit_date,
           count(distinct doctor_id) as num_doctors_seen
    from {s}.visit group by patient_id
), drugs as (
    select v.patient_id, p.drug_name, sum(p.quantity) as quantity
    from {s}.prescription p join {s}.visit v on v.visit_id = p.visit_id
    group by v.patient_id, p.drug_name
), totals as (
    select patient_id, cast(sum(quantity) as bigint) as total_drug_quantity from drugs
    group by patient_id
), top_drug as (
    select patient_id, drug_name from (
        select patient_id, drug_name,
               row_number() over (partition by patient_id order by quantity desc, drug_name asc) as rn
        from drugs
    ) ranked where rn = 1
)
select p.patient_id,
       p.full_name,
       coalesce(stats.num_visits, 0) as num_visits,
       strftime(stats.first_visit_date, '%Y-%m-%d') as first_visit_date,
       coalesce(stats.num_doctors_seen, 0) as num_doctors_seen,
       coalesce(totals.total_drug_quantity, 0) as total_drug_quantity,
       top_drug.drug_name as most_prescribed_drug
from {s}.patient p
left join stats on stats.patient_id = p.patient_id
left join totals on totals.patient_id = p.patient_id
left join top_drug on top_drug.patient_id = p.patient_id
""",
        ),
    ),  # fmt: skip
    generate=_gen_clinic,
)


# ---------------------------------------------------------------------------------
# airfare
# ---------------------------------------------------------------------------------
def _gen_airfare(rng: random.Random) -> dict[str, Rows]:
    airlines = [
        {"airline_code": code, "airline_name": name}
        for code, name in [
            ("AX", "Apex Air"), ("BL", "Blue Lark"), ("CQ", "Coastal Quest"),
            ("DN", "Delta North"), ("EV", "Evergreen"), ("FZ", "Frontier Zephyr"),
        ]  # fmt: skip
    ]
    airports = [
        {"airport_code": code, "city": city, "state": state}
        for code, city, state in [
            ("ABQ", "Albuquerque", "NM"), ("BOS", "Boston", "MA"), ("CLT", "Charlotte", "NC"),
            ("DEN", "Denver", "CO"), ("ELP", "El Paso", "TX"), ("FAT", "Fresno", "CA"),
            ("GEG", "Spokane", "WA"), ("HNL", "Honolulu", "HI"), ("IND", "Indianapolis", "IN"),
            ("JAX", "Jacksonville", "FL"),
        ]  # fmt: skip
    ]
    # FZ has no flights; JAX has no traffic.
    carrier_codes = [a["airline_code"] for a in airlines[:-1]]
    busy = [a["airport_code"] for a in airports[:-1]]
    weights = [6, 5, 5, 4, 3, 2, 2, 1, 1]
    flights = []
    for flight_id in range(1, 601):
        origin = rng.choices(busy, weights)[0]
        destination = rng.choices([c for c in busy if c != origin])[0]
        cancelled = 1 if rng.random() < 0.06 else 0
        flights.append(
            {
                "flight_id": flight_id,
                "airline_code": rng.choice(carrier_codes),
                "origin": origin,
                "destination": destination,
                "flight_date": _date(rng),
                "dep_delay": None if cancelled else rng.randint(-10, 120),
                "cancelled": cancelled,
            }
        )
    return {"flight": flights, "airline": airlines, "airport": airports}


AIRFARE = TaskSpec(
    name="airfare",
    tables=(
        SourceTable("flight", "sqlite", (
            Column("flight_id", "the unique identifier of the flight"),
            Column("airline_code", "the code of the operating airline"),
            Column("origin", "the code of the departure airport"),
            Column("destination", "the code of the arrival airport"),
            Column("flight_date", "the date of the flight, YYYY-MM-DD"),
            Column("dep_delay", "departure delay in minutes; negative means early; NULL for cancelled flights"),
            Column("cancelled", "1 if the flight was cancelled, otherwise 0"),
        )),
        SourceTable("airline", "file", (
            Column("airline_code", "the unique code of the airline"),
            Column("airline_name", "the name of the airline"),
        ), fmt="csv"),
        SourceTable("airport", "documents", (
            Column("airport_code", "the unique code of the airport"),
            Column("city", "the city of the airport"),
            Column("state", "the state of the airport"),
        )),
    ),  # fmt: skip
    models=(
        DataModel(
            name="airlines",
            description="Each record represents an airline.",
            key=("airline_code",),
            columns=(
                Column("airline_code", "Unique code of the airline."),
                Column("airline_name", "Name of the airline."),
                Column("num_flights", "Number of flights operated by the airline, including cancelled ones, replace NULL values with 0."),
                Column("num_cancelled", "Number of cancelled flights of the airline, replace NULL values with 0."),
                Column("pct_cancelled", "Percentage (0-100) of the airline's flights that were cancelled, rounded to 4 decimal places. NULL if the airline has no flights."),
                Column("avg_dep_delay", "Average departure delay in minutes over the airline's flights that were not cancelled, rounded to 4 decimal places. NULL if there are none."),
                Column("top_destination", "Code of the destination airport with the most flights of the airline (including cancelled ones), with ties broken by the ascending order of the airport code. NULL if the airline has no flights."),
            ),
            sql="""
with stats as (
    select airline_code,
           count(*) as num_flights,
           cast(sum(cancelled) as bigint) as num_cancelled,
           100.0 * sum(cancelled) / count(*) as pct_cancelled,
           avg(case when cancelled = 0 then dep_delay end) as avg_dep_delay
    from {s}.flight group by airline_code
), dest as (
    select airline_code, destination from (
        select airline_code, destination,
               row_number() over (
                   partition by airline_code order by count(*) desc, destination asc
               ) as rn
        from {s}.flight group by airline_code, destination
    ) ranked where rn = 1
)
select a.airline_code,
       a.airline_name,
       coalesce(stats.num_flights, 0) as num_flights,
       coalesce(stats.num_cancelled, 0) as num_cancelled,
       round(stats.pct_cancelled, 4) as pct_cancelled,
       round(stats.avg_dep_delay, 4) as avg_dep_delay,
       dest.destination as top_destination
from {s}.airline a
left join stats on stats.airline_code = a.airline_code
left join dest on dest.airline_code = a.airline_code
""",
        ),
        DataModel(
            name="airports",
            description="Each record represents an airport.",
            key=("airport_code",),
            columns=(
                Column("airport_code", "Unique code of the airport."),
                Column("city", "City of the airport."),
                Column("num_departures", "Number of flights departing from the airport, including cancelled ones, replace NULL values with 0."),
                Column("num_arrivals", "Number of flights arriving at the airport, including cancelled ones, replace NULL values with 0."),
                Column("num_airlines_departing", "Number of distinct airlines with flights departing from the airport, replace NULL values with 0."),
                Column("busiest_departure_date", "Date (format YYYY-MM-DD) with the most departing flights, with ties broken by the earliest date. NULL if the airport has no departures."),
                Column("is_above_avg_departures", "Set to 1 if the airport's number of departures is greater than the average number of departures over all airports (airports without departures count as 0); otherwise, set to 0."),
            ),
            sql="""
with dep as (
    select origin as airport_code, count(*) as num_departures,
           count(distinct airline_code) as num_airlines
    from {s}.flight group by origin
), arr as (
    select destination as airport_code, count(*) as num_arrivals
    from {s}.flight group by destination
), busiest as (
    select airport_code, flight_date from (
        select origin as airport_code, cast(flight_date as date) as flight_date,
               row_number() over (
                   partition by origin order by count(*) desc, cast(flight_date as date) asc
               ) as rn
        from {s}.flight group by origin, cast(flight_date as date)
    ) ranked where rn = 1
), base as (
    select a.airport_code, a.city,
           coalesce(dep.num_departures, 0) as num_departures,
           coalesce(arr.num_arrivals, 0) as num_arrivals,
           coalesce(dep.num_airlines, 0) as num_airlines_departing,
           strftime(busiest.flight_date, '%Y-%m-%d') as busiest_departure_date
    from {s}.airport a
    left join dep on dep.airport_code = a.airport_code
    left join arr on arr.airport_code = a.airport_code
    left join busiest on busiest.airport_code = a.airport_code
)
select base.*,
       case when num_departures > (select avg(num_departures) from base) then 1 else 0 end
           as is_above_avg_departures
from base
""",
        ),
    ),  # fmt: skip
    generate=_gen_airfare,
)


SPECS: dict[str, TaskSpec] = {spec.name: spec for spec in (BOOKSHOP, CLINIC, AIRFARE)}
