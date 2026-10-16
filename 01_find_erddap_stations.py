import os
import re
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from io import StringIO
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from erddapy import ERDDAP
from tqdm import tqdm
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



######################################
# Adaptive Per-Server Throttle
######################################

class AdaptiveServerThrottle:
    """
    Per-server concurrency limiter that automatically reduces concurrency
    when a server rejects requests (rate-limiting).

    Each server starts with `initial_slots` concurrent request slots.
    On rejection, one slot is permanently removed (minimum `min_slots`).
    """

    def __init__(self, initial_slots=5, min_slots=1):
        self._lock = threading.Lock()
        self._initial = initial_slots
        self._min = min_slots
        self._semaphores = {}       # server_url -> Semaphore
        self._current_slots = {}    # server_url -> int

    def _ensure_server(self, url):
        with self._lock:
            if url not in self._semaphores:
                self._semaphores[url] = threading.Semaphore(self._initial)
                self._current_slots[url] = self._initial

    @contextmanager
    def request_slot(self, url):
        """Acquire a per-server slot; releases on exit."""
        self._ensure_server(url)
        self._semaphores[url].acquire()
        try:
            yield
        finally:
            self._semaphores[url].release()

    def reduce_concurrency(self, url):
        """Permanently remove one slot for this server (acquire without release)."""
        with self._lock:
            current = self._current_slots.get(url, self._initial)
            if current <= self._min:
                return current
        # Try non-blocking acquire outside the lock to avoid deadlock
        acquired = self._semaphores[url].acquire(blocking=False)
        if acquired:
            with self._lock:
                new = self._current_slots[url] - 1
                self._current_slots[url] = new
            logging.warning(f"Reduced concurrency for {url} to {new}")
            return new
        return self._current_slots.get(url, self._initial)

    def get_current_slots(self, url):
        with self._lock:
            return self._current_slots.get(url, self._initial)

    def summary(self):
        """Return dict of servers whose concurrency was reduced."""
        with self._lock:
            return {url: slots for url, slots in self._current_slots.items()
                    if slots < self._initial}


def _read_csv_with_timeout(url, timeout=30):
    """Read CSV from URL with explicit timeout (pd.read_csv has no timeout)."""
    response = urlopen(url, timeout=timeout)
    text = response.read().decode('utf-8')
    return pd.read_csv(StringIO(text))


######################################
# Functions Define Section Start
######################################

#--------------------------
# Function to process search ERDDAP servers with parallelizing 30 threads
#--------------------------
def search_erddap_servers(erddap_links, min_time, max_time, step1_result_filename, step1_force_rerun):
    pickle_file = step1_result_filename #"initial_erddap_stations.pkl"
    if os.path.exists(pickle_file) and not step1_force_rerun:
        try:
            FF = pd.read_pickle(pickle_file)
            print(f"Loading existing DataFrame from {pickle_file}")
        except:
            print(f"Error loading {pickle_file}, re-running processing")
            # Proceed to processing
    else:
        kw = {"min_time": min_time, "max_time": max_time}

        def process_erddap_server(org):
            erddap_object = ERDDAP(server=org, protocol="tabledap", response="csv")
            url = erddap_object.get_search_url(search_for='', response="csv", **kw)
            try:
                dataset_ids = pd.read_csv(url)["Dataset ID"].unique()
                return {"org": org, "dataset_ids": dataset_ids}
            except:
                return {"org": org, "dataset_ids": []}

        allsites = []
        allurl = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            results = list(tqdm(
                executor.map(process_erddap_server, erddap_links),
                total=len(erddap_links),
                desc="Searching ERDDAP servers"
            ))

        for result in results:
            dataset_ids = result["dataset_ids"]
            org = result["org"]
            if len(dataset_ids) > 0:
                allsites.extend(dataset_ids)
                allurl.extend([org] * len(dataset_ids))

        FF = pd.DataFrame({"sites": allsites, "url": allurl})
        FF["time_coverage_start"] = np.nan
        FF["time_coverage_end"] = np.nan
        FF["geospatial_lat_min"] = np.nan
        FF["geospatial_lat_max"] = np.nan
        FF["geospatial_lon_min"] = np.nan
        FF["geospatial_lon_max"] = np.nan
        FF["check_sum"] = 1

        FF.to_pickle(pickle_file)
        csv_file = os.path.splitext(pickle_file)[0] + ".csv"
        FF.to_csv(csv_file, index=False)
        print(f"Saved DataFrame to {pickle_file} and {csv_file}")

    return FF


#--------------------------
# Function to do metadata processing by parallelizing the process of each single ERDDAP site
# Added data type labeling function
#--------------------------
def process_metadata(FF, min_time, max_time, searchterms, keywords, step2_result_filename, step2_force_rerun):
    results_file = step2_result_filename
    if os.path.exists(results_file) and not step2_force_rerun:
        try:
            print(f"Loading existing results from {results_file}")
            FF = pd.read_pickle(results_file)
        except:
            print(f"Error loading {results_file}, re-running processing")
            # Proceed to processing
    else:
        # Initialize data type columns
        for kwd in keywords:
            FF[kwd] = False
        FF["check_sum"] = np.nan

        throttle = AdaptiveServerThrottle(initial_slots=5, min_slots=1)

        def process_site(jj, url, site, min_time, max_time, searchterms, keywords, max_retries=3):
            result = {"index": jj, "check_sum": np.nan}
            for kwd in keywords:
                result[kwd] = False

            for attempt in range(max_retries):
                try:
                    with throttle.request_slot(url):
                        e = ERDDAP(server=url, protocol="tabledap", response="csv")
                        e.dataset_id = site
                        e.constraints = {"time>=": min_time, "time<=": max_time}
                        info_url = e.get_info_url()

                        df = _read_csv_with_timeout(info_url, timeout=30)

                        # Check searchterms in Variable Name
                        idx = df["Variable Name"].str.contains(searchterms, flags=re.IGNORECASE, na=False)
                        if not idx.any():
                            return result  # No match -- not a failure, just no relevant data

                        # Reset constraints for full metadata
                        e.constraints = {}
                        info_url = e.get_info_url()
                        df = _read_csv_with_timeout(info_url, timeout=30)

                        result["check_sum"] = 1
                        # Extract metadata
                        for attr in ["time_coverage_start", "time_coverage_end",
                                     "geospatial_lat_min", "geospatial_lat_max",
                                     "geospatial_lon_min", "geospatial_lon_max"]:
                            try:
                                result[attr] = df.loc[df['Attribute Name'] == attr, 'Value'].item()
                            except (ValueError, KeyError):
                                pass

                        # Check keywords in Value
                        for kwd in keywords:
                            if df["Value"].str.contains(kwd, flags=re.IGNORECASE, na=False).any():
                                result[kwd] = True

                        return result  # Success

                except Exception as e:
                    logging.warning(f"[{site}@{url}] attempt {attempt+1}/{max_retries}: {e}")
                    throttle.reduce_concurrency(url)

                # Backoff OUTSIDE the slot so the slot is released for other tasks
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

            logging.error(f"[{site}@{url}] all {max_retries} attempts failed")
            return result

        with ThreadPoolExecutor(max_workers=30) as executor:
            args_list = [(jj, FF['url'][jj], FF['sites'][jj], min_time, max_time, searchterms, keywords) for jj in range(len(FF))]
            results = list(tqdm(executor.map(lambda x: process_site(*x), args_list), total=len(FF)))

        for result in results:
            jj = result.pop("index")
            for key, value in result.items():
                FF.loc[jj, key] = value

        FF = FF.dropna(subset="check_sum")
        FF.reset_index(drop=True, inplace=True)
        FF = FF.drop(columns="check_sum")

        FF["geospatial_lat_min"] = pd.to_numeric(FF["geospatial_lat_min"], errors='coerce')
        FF["geospatial_lon_min"] = pd.to_numeric(FF["geospatial_lon_min"], errors='coerce')
        FF["geospatial_lat_max"] = pd.to_numeric(FF["geospatial_lat_max"], errors='coerce')
        FF["geospatial_lon_max"] = pd.to_numeric(FF["geospatial_lon_max"], errors='coerce')

        FF.to_pickle(results_file)
        FF_save = FF.copy()
        labeled_csv = os.path.join(os.path.dirname(results_file), 'erddap_station_search_results_labeled.csv')
        FF_save.to_csv(labeled_csv, index=False)
        print(f"Saved results to {results_file} and {labeled_csv}")

        # Summarize data types
        for kwd in keywords:
            print(f"{kwd}: {len(FF[FF[kwd] == True])} datasets")

        # Log which servers had their concurrency reduced
        reduced = throttle.summary()
        if reduced:
            logging.info("Servers with reduced concurrency (initial=5):")
            for srv, slots in sorted(reduced.items()):
                logging.info(f"  {srv}: reduced to {slots} slot(s)")

    return FF
    
######################################
# Functions Define Section End
######################################


######################################
# Script Interface
######################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Search and process ERDDAP stations with optional rerun")
    parser.add_argument('--step1-force-rerun', action='store_true', help="Force re-running the ERDDAP server search")
    parser.add_argument('--step2-force-rerun', action='store_true', help="Force re-running the metadata processing")
    parser.add_argument('--erddaps-json', default=os.path.join(HERE, 'config', 'erddaps.json'),
                        help="ERDDAP server list (from IrishMarineInstitute/awesome-erddap)")
    parser.add_argument('--results-dir', default=os.path.join(HERE, 'results'),
                        help="Where the step-1/step-2 pickles and CSVs are written")
    args = parser.parse_args()
    
    # Load a list of all the ERDDAPs and their urls.
    # This is saved as a JSON with the code bc I cant get the direct link to the git site to work.
    os.makedirs(args.results_dir, exist_ok=True)
    ERDDAPs_file = open(args.erddaps_json)
    data_json = pd.read_json(ERDDAPs_file)
    erddap_links = data_json['url'].unique()
    print('erddap_links',erddap_links)
    
    # Define parameters
    min_time = "2018-10-01T00:00:00Z"
    max_time = "2026-01-01T00:00:00Z"
    
    # Define search terms
    searchterms = r"chl|turb|sedi|par|phot|secchi|ssc"
    
    # Define keywords for identify which type of data is present in each dataset
    keywords = {'buoy', 'glider', 'ship', 'turbid', 'ssc', 'photo', 'chl', 'radiation'}
    
    # Define the filenames for two step results 
    Step1_ResultFilename = os.path.join(args.results_dir, 'initial_erddap_stations.pkl')
    Step2_ResultFilename = os.path.join(args.results_dir, 'Saved_all_relevant_stations.pkl')
    
    # Run search section
    FF = search_erddap_servers(erddap_links, min_time, max_time, Step1_ResultFilename, args.step1_force_rerun)

    # Run combined metadata processing and data type labeling
    FF = process_metadata(FF, min_time, max_time, searchterms, keywords, Step2_ResultFilename, args.step2_force_rerun)

    print(f"Total matching datasets: {len(FF)}")

