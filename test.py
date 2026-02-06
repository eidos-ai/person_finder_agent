import asyncio
import time

from loguru import logger

from agent import PersonInitialDescription, run_agent

PERSONS = [
    PersonInitialDescription("TINDRA", "AAKERBLOM", "N Hollywood", "CA", "Santa Monica College", "CSU General Education", "07/01/2025"),
    # PersonDescription("TAMIKA", "AARON", "Los Angeles", "CA", "Santa Monica College", "Marketing", "07/01/2025"),
    # PersonDescription("BENJAMIN", "AASHEIM", "Los Angeles", "CA", "Santa Monica College", "Computer Science", "07/03/2025"),
    # PersonDescription("FRANCES", "ABANGAN", "Los Angeles", "CA", "Santa Monica College", "General Science", "07/01/2025"),
    # PersonDescription("AMINA", "ABASOVA", "Los Angeles", "CA", "Santa Monica College", "IGETC General Education", "06/26/2025"),
    # PersonDescription("RAMTIN", "ABAYAHOUDAIAN", "Beverly Hills", "CA", "Santa Monica College", "General Science", "07/01/2025"),
    # PersonDescription("MOHAMED", "ABBAD", "Los Angeles", "CA", "Santa Monica College", "Cloud Computing", "06/27/2025"),
    # PersonDescription("SAMAR", "ABBASHER", "Los Angeles", "CA", "Santa Monica College", "Nature-based Pedagogy", "07/01/2025"),
    # PersonDescription("SEAN", "ABBOTT", "Los Angeles", "CA", "Santa Monica College", "Liberal Arts - Social & Behavioral Science", "07/01/2025"),
    # PersonDescription("MAIVELLE", "ABD EL MISSH", "Lawndale", "CA", "Santa Monica College", "General Science", "07/01/2025"),
    # PersonDescription("WINDY", "ABDELMALAK", "Inglewood", "CA", "Santa Monica College", "General Science", "07/01/2025"),
]


async def process_person(person: PersonInitialDescription) -> tuple[float, float]:
    name = f"{person.first_name} {person.last_name}"
    start = time.time()
    total_cost = 0.0

    try:
        result = await run_agent(person)

        logger.info("RESULT: {}\n{}", name, result.profile)
        logger.info("Final Response: {}\n{}", name, result.response)

        c = result.consumption
        for model, usage in c.models.items():
            cost = usage.cost_usd
            total_cost += cost
            logger.info(
                "{} | in={} | out={} | thinking={} | cost=${:.6f}",
                model, usage.input_tokens, usage.output_tokens,
                usage.thinking_tokens, cost,
            )
        logger.info("Tool calls: {}", c.tool_calls_used)

    except Exception as e:
        logger.exception(e)
        logger.error("{} -> ERROR: {}", name, e)

    elapsed = time.time() - start
    logger.info("{} completed in {:.1f}s", name, elapsed)
    return total_cost, elapsed


async def main():
    total_cost = 0.0
    total_elapsed = 0.0

    for person in PERSONS:
        cost, elapsed = await process_person(person)
        total_cost += cost
        total_elapsed += elapsed

    logger.info(
        "TOTAL COST USD: ${:.6f} | TIME: {:.1f}s | PERSONS: {}",
        total_cost, total_elapsed, len(PERSONS),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
