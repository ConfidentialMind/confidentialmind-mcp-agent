import websockets
import asyncio
from threading import Lock, Thread
import os
from pydantic import BaseModel, ValidationError
from typing import Literal, Optional, List, Union, Dict
import json
import requests
import datetime
import re
from dotenv import load_dotenv

from . import config

configs_read = False


def load_environment() -> None:
    """
    Load environment variables from a .env file if it exists.
    Sets the following config object variables:
    - LOCAL_DEV
    - LOCAL_CONFIGS
    """
    global configs_read

    if configs_read:
        return
    env_filename = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_filename):
        load_dotenv(os.path.join(os.getcwd(), ".env"), override=True)
        config.LOCAL_DEV = True

    if os.environ.get("CONFIDENTIAL_MIND_LOCAL_CONFIG") == "True":
        config.LOCAL_CONFIGS = True

    configs_read = True
    print(
        f"Done loading environment configs: \n \
        LOCAL_DEV: {config.LOCAL_DEV} \n \
        LOCAL_CONFIGS: {config.LOCAL_CONFIGS}"
    )


def get_api_parameters(config_id: str, index: Optional[int] = None) -> tuple[str, Optional[dict]]:
    """
    Construct and return the URL associated with a given configuration ID based on its stack ID and type.
    In case of .env file values from the file are used.
    It handles different environments, such as local development and production.
    For array connectors, the index parameter specifies which connector in the array to use.

    Args:
        config_id (str): The identifier for the configuration.
        index (Optional[int]): For array connectors, the index of the connector to use.

    Returns:
        tuple: A tuple containing the base URL (str) and a header dictionary with api-key (dict).
               A dictionary for request headers in the format {"Authorization": "Bearer api-key-value"}. 
               The API key is only used in local development environments.
    """
    load_environment()
    if config.LOCAL_DEV:
        # For array connectors with an index
        if index is not None:
            url_var = f"{config_id}_{index}_URL"
            apikey_var = f"{config_id}_{index}_APIKEY"
        else:
            url_var = f"{config_id}_URL"
            apikey_var = f"{config_id}_APIKEY"
        
        url_base = os.environ.get(url_var, None)
        # remove the trailing / if it exists with regex
        if url_base is not None:
            url_base = re.sub(r"/$", "", url_base)
        apikey = os.environ.get(apikey_var, None)
        headers = {}
        # check that the api key is valid JWT if it exists
        if apikey is not None:
            try:
                if not re.match(
                    r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$", apikey
                ):
                    raise ValueError
            except ValueError:
                raise ValueError(f"API key for {url_var} is not the right format")
            headers = {"Authorization": f"Bearer {apikey}"}
    else:
        configManager = ConfigManager()
        # For array connectors
        if index is not None:
            urls = configManager.getUrlForConnector(config_id)
            if isinstance(urls, list) and 0 <= index < len(urls):
                url_base = urls[index]
            else:
                raise ValueError(f"Invalid index {index} for array connector {config_id}")
        else:
            url_base = configManager.getUrlForConnector(config_id)
        headers = {}  # not used

    return (url_base, headers)

def get_array_api_parameters(config_id: str) -> list[tuple[str, Optional[dict]]]:
    """
    Get all URLs and headers for an array connector in local development mode.
    
    Args:
        config_id (str): The identifier for the configuration.
        
    Returns:
        list[tuple]: A list of tuples, each containing a base URL and headers.
    """
    load_environment()
    results = []
    
    if config.LOCAL_DEV:
        # Check for indexed environment variables
        index = 0
        max_index = 10  # Safety limit to prevent infinite loops
        
        while index < max_index:
            url_var = f"{config_id}_{index}_URL"
            if os.environ.get(url_var) is None:
                break
                
            url = os.environ.get(url_var)
            
            # remove the trailing / if it exists with regex
            if url is not None:
                url = re.sub(r"/$", "", url)
            
            # Get API key for this index
            apikey_var = f"{config_id}_{index}_APIKEY"
            apikey = os.environ.get(apikey_var)
            headers = {}
            
            # Check that the API key is valid JWT if it exists
            if apikey is not None:
                try:
                    if not re.match(
                        r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$", apikey
                    ):
                        print(f"API key for {apikey_var} is not the right format")
                    else:
                        headers = {"Authorization": f"Bearer {apikey}"}
                except Exception as e:  # Catch any exception to prevent infinite loop
                    print(f"Error validating API key for {apikey_var}: {e}")
                    # Don't raise, continue to next index
            
            results.append((url, headers))
            index += 1
        
        # Log if we hit the safety limit
        if index >= max_index:
            print(f"Reached maximum index {max_index} for {config_id} array connector variables")
            
        # If no indexed variables found, try to use ConfigManager to get array URLs
        if not results:
            try:
                if not config.LOCAL_CONFIGS:
                    print(f"No array variables found for {config_id}, trying ConfigManager")
                    configManager = ConfigManager()
                    urls = configManager.getUrlForConnector(config_id)
                    if isinstance(urls, list):
                        results = [(url, {}) for url in urls]
                else:
                    print(f"LOCAL_CONFIGS is True, skipping ConfigManager for {config_id}")
            except Exception as e:
                print(f"Failed to get array URLs from ConfigManager: {e}")
                # Don't raise, continue with empty results
    else:
        # Use ConfigManager to get array URLs
        try:
            print(f"Not in LOCAL_DEV mode, using ConfigManager for {config_id}")
            configManager = ConfigManager()
            urls_and_names = configManager.getUrlAndNameForConnector(config_id)
            
            if isinstance(urls_and_names, list):
                for item in urls_and_names:
                    url = item.get("url")
                    if url:
                        results.append((url, {}))
            elif isinstance(urls_and_names, dict) and "url" in urls_and_names:
                # Single result returned as dict
                results.append((urls_and_names["url"], {}))
        except Exception as e:
            print(f"Error getting URLs from ConfigManager: {e}")
            # Don't raise, continue with empty results
    
    print(f"Found {len(results)} endpoints for {config_id}")
    return results

# Type definition for all possible connector types
ConnectorType = Literal[
    "bucket",
    "database",
    "endpoint",
    "agent_endpoint",
    "model_endpoint", 
    "data_source",
    "agent_tool",
    "api",
    "llm",
    "model",
    "external_llm",
    "external_database",
    "playground",
    "frontend",
    "volume"
]

class ConnectorSchema(BaseModel):
    """
    A schema representing a connector. A connector connects two services running inside the stack.

    Attributes:
        type (ConnectorType): The type of the service to be connected.
        label (str): Label to be shown in admin UI.
        config_id (str): Id that can be used in code to create connection class.
        stack_id (str): The unique identifier of the resource.
        deployment_name (str): The name of the deployment.
    """
    type: ConnectorType
    label: str
    config_id: str
    stack_id: Optional[str] = None
    deployment_name: Optional[str] = None

class ArrayConnectorSchema(BaseModel):
    """
    A schema representing an array connector that can connect to multiple services.

    Attributes:
        type (ConnectorType): The type of the service to be connected.
        label (str): Label to be shown in admin UI.
        config_id (str): Id that can be used in code to create connection class.
        stack_ids (list[str]): List of unique identifiers of the resources.
        deployment_names (list[str]): List of names of the deployments with 1:1 mapping with stack_ids.
    """
    type: ConnectorType
    label: str
    config_id: str
    stack_ids: Optional[List[str]] = None
    deployment_names: Optional[List[str]] = None

class ConnectorsDBSchema(BaseModel):
    """A schema encapsulating lists of both regular and array connectors."""
    connectors: list[ConnectorSchema]
    array_connectors: Optional[list[ArrayConnectorSchema]] = None


class ConfigPatch(BaseModel):
    # TODO should this be optional?
    config: Optional[dict] = None
    connectors: Optional[ConnectorsDBSchema] = None
    config_schema: Optional[dict] = None
    deployment_state: Literal["deployed"] = "deployed"


def get_namespace_for_type(connector_type: str) -> str:
    """
    Maps connector types to their corresponding namespaces.

    Args:
        connector_type (str): The type of connector.
        
    Returns:
        str: The corresponding namespace.
    """
    namespace_mapping = {
        "bucket": "rook-ceph",
        "database": "databases",
        "endpoint": "api-services",
        "agent_endpoint": "api-services", 
        "model_endpoint": "api-services",
        "data_source": "api-services",
        "agent_tool": "api-services",
        "api": "api-services",
        "llm": "api-services",
        "model": "api-services",
        "external_llm": "api-services",
        "external_database": "api-services",
        "playground": "tools",
        "frontend": "frontend",
        "volume": "api-services"
    }
    return namespace_mapping.get(connector_type, "default")


class SingletonClass(object):
    _lock = Lock()

    def __new__(cls):
        # two checks are on purpose, check-lock-check pattern
        if not hasattr(cls, "instance"):
            with cls._lock:
                if not hasattr(cls, "instance"):
                    print("Creating new instance")
                    cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance

    # TODO come up with way to check if instance is initialized


# Singleton class for managing the configuration of the application
class ConfigManager(SingletonClass):
    """
    The ConfigManager class handles configuration management between the application and the server,
    including fetching and updating configurations. It uses websockets for real-time updates
    and HTTP requests for initial fetches or patches to the configuration manager.

    Attributes:
        __manager_url (str): The URL of the configuration manager.
        __config_model (BaseModel): The model representing the app config schema.
        __lock (RLock): A lock used for thread-safe operations on configurations.
        __initialized (bool): A flag indicating whether the manager is initialized.
        __connectors (list[ConnectorSchema]): Current list of connectors managed by the application.
        __array_connectors (list[ArrayConnectorSchema]): Current list of array connectors.
        __config (BaseToolConfig | None): The current configuration object.
    """

    def init_manager(
        self,
        config_model: Optional[BaseModel],
        connectors: Optional[list[ConnectorSchema]] = None,
        array_connectors: Optional[list[ArrayConnectorSchema]] = None,
    ):
        """
        Initializes the configuration manager. This method should be called before any other operations
        to ensure that configurations are properly fetched and initialized.

        Args:
            config_model (BaseModel, optional): The app config model.
            connectors (list[ConnectorSchema], optional): List of connectors the app may use.
            array_connectors (list[ArrayConnectorSchema], optional): List of array connectors the app may use.
        """
        # TODO should initialized be a state with initalizing variable if multiple FEs are open
        # TODO this is most likely issue only for local development with multiple frontends open
        print("Initializing config manager")
        try:
            # TODO this should use lock?
            if self.__initialized:
                print("Config manager already initialized")
                return
        except AttributeError:
            # Setting initiliazed to True immediately to prevent multiple init calls
            self.__initialized = True

        # initialize the environment variables
        load_environment()
        # Always use env variable if it exists, prevents accidental use of old id
        # TODO should we remove the ID parameter?
        self.__id = os.environ.get("SERVICE_NAME")
        self.__manager_url = None
        self.__realtime_url = None

        # id is needed if we don't use local configs
        assert not (
            self.__id is None and config.LOCAL_CONFIGS is False
        ), "Service name (id) must be set in env variable SERVICE_NAME if not using local configs"

        if config.LOCAL_DEV:
            # try to read from env variables
            # TODO move these to environment initialization?
            if os.environ.get("MANAGER_URL") is not None:
                print("Using manager url from env")
                self.__manager_url = (
                    os.environ.get("MANAGER_URL") + f"/internal/{self.__id}"
                )
            if os.environ.get("REALTIME_URL") is not None:
                print("Using realtime url from env")
                self.__realtime_url = (
                    os.environ.get("REALTIME_URL") + f"/internal/{self.__id}"
                )

        # if not in local dev, use the default urls
        if self.__manager_url is None:
            self.__manager_url = f"{config.MANAGER_SERVICE_URL}/internal/{self.__id}"
        if self.__realtime_url is None:
            self.__realtime_url = f"{config.REALTIME_SERVICE_URL}/internal/{self.__id}"

        self.__lock = Lock()
        self.__config_model = config_model
        # TODO should connectors be defaulted to empty array in sql
        self.__connectors = None
        self.__array_connectors = None
        self.__config = None

        if config.LOCAL_CONFIGS:
            if connectors is None and array_connectors is None:
                temp_connectors = None
            else:
                temp_connectors = ConnectorsDBSchema(
                    connectors=connectors or [],
                    array_connectors=array_connectors or []
                )

            if config_model is None:
                temp_model = None
            else:
                temp_model = config_model.model_dump()

            self.__set_configs({"config": temp_model, "connectors": temp_connectors})
        else:
            # Fetch initial config
            self.__fetch_config(initial_request=True)

            # Update config with new connectors
            self.__update_config(config_model, connectors, array_connectors)

            # Fetch updated configs
            self.__fetch_config()

        # After this we should have all the configs, error if not?
        # TODO check if config is None still?

        self.__close_sockets = False

        # Create a thread and loop for websocket
        # Loop is for using asyncio in the thread (required for the websockets library)
        # Daemon thread is automatically closed when main thread is closed
        if not config.LOCAL_CONFIGS:
            loop = asyncio.new_event_loop()
            thread = Thread(target=self.__start_loop, args=(loop,), daemon=True)
            thread.start()
            asyncio.run_coroutine_threadsafe(self.__run_websocket_for_ever(), loop)

        print("setting initialized")
        self.__initialized = True

    # Wrapper for websocket thread
    def __start_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    # Protect so that can be running only once
    async def __run_websocket_for_ever(self):
        print("Starting websocket connection")
        # TODO do we want to wait for succesful connection before letting the application start?
        sleep = 0
        while not self.__close_sockets:
            try:
                # TODO should we fetch configs separately if websocket fails to catch missed notifications
                # max sleep time is 5 seconds
                if sleep < 5:
                    sleep += 1

                print(f"{datetime.datetime.now()} - Starting websocket connection")
                async with websockets.connect(self.__realtime_url) as websocket:
                    # Reset sleep time after succesful connection
                    sleep = 0
                    print(f"{datetime.datetime.now()} - New websocket connection")
                    # Process messages received on the connection.
                    async for message in websocket:
                        # TODO test only connector patch from websocket, does it have config in addition to connectors?
                        parsed = json.loads(message)
                        configs = parsed["payload"]
                        self.__set_configs(configs)
                        if configs:
                            # Refetch the config to get displaynames
                            self.__fetch_config()

            except websockets.ConnectionClosed as e:
                print(f"{datetime.datetime.now()} - Exception from websocket connection closed")
                print(e)
                await asyncio.sleep(sleep)
                continue
            except ConnectionRefusedError as e:
                print(f"{datetime.datetime.now()} - ConnectionRefused, likely the server is not running")
                print(e)
                await asyncio.sleep(sleep)
                continue
            except Exception as e:
                print(f"{datetime.datetime.now()} - Exception from websocket")
                print(e)
                await asyncio.sleep(sleep)
                continue
        print(f"{datetime.datetime.now()} - Exited websocket setup")

    # This is blocking request on purpose!
    def __fetch_config(self, initial_request=False):
        # TODO for local development it is possible to get non-entry case
        response = requests.get(self.__manager_url)
        self.__set_configs(response.json(), initial_request)

    def __set_configs(self, json_configs, initial_request=False):
        try:
            # print(json_configs)
            if initial_request:
                config = json_configs["config"]
                config_schema = json_configs["config_schema"]
                if config is None and config_schema is None:
                    print("No config or config schema found, skipping")
                    return

            connectors_data = json_configs.get("connectors")
            if connectors_data is not None:
                connectors_config = ConnectorsDBSchema.model_validate(connectors_data)
            else:
                connectors_config = None

            if json_configs.get("config") is not None:
                config = self.__config_model.model_validate(json_configs["config"])
            else:
                config = None

            # lock makes sure that websocket thread and db funcs dont run at the same time
            with self.__lock:
                if connectors_config is not None:
                    self.__connectors = connectors_config.connectors
                    self.__array_connectors = connectors_config.array_connectors
                    print("Successfully set connectors and array connectors")
                if config is not None:
                    self.__config = config
                    print("Successfully set configs")

        except (ValidationError, ValueError) as e:
            print("Failed to validate configs")
            print(e)
            return

    def __update_config(
        self,
        config_model: BaseModel,
        connectors: Optional[list[ConnectorSchema]] = None,
        array_connectors: Optional[list[ArrayConnectorSchema]] = None,
    ):
        update_body = ConfigPatch()

        if connectors is not None or array_connectors is not None:
            update_body.connectors = ConnectorsDBSchema(
                connectors=connectors or [],
                array_connectors=array_connectors or []
            )

        if config_model is not None:
            # Always send the latest schema
            update_body.config_schema = config_model.model_json_schema()
            
            if self.__config is None:
                # If config is None, we set it directly from the model
                update_body.config = config_model.model_dump()
            else:
                try:
                    # Let Pydantic handle the parsing - it will ignore extra fields and use defaults
                    existing_data = self.__config.model_dump() if hasattr(self.__config, 'model_dump') else self.__config
                    parsed_config = config_model.model_validate(existing_data)
                    update_body.config = parsed_config.model_dump()
                except Exception:
                    update_body.config = config_model.model_dump()

        response = requests.patch(self.__manager_url, json=update_body.model_dump())
        if response.status_code >= 400:
                try:
                    error_data = response.json()
                    print(f"Request failed with status {response.status_code}")
                    print(f"Error: {error_data.get('error', 'Unknown error')}")
                    print(f"Details: {error_data.get('detail', 'No details provided')}")
                    print(f"Request payload: {json.dumps(update_body.model_dump(), indent=2)}")
                except Exception as e:
                    print(f"Failed to parse error response: {e}")
                    print(f"Request failed with status {response.status_code}, but couldn't parse error response")
                    print(f"Response text: {response.text}")
        else:
            print(f"Request succeeded with status {response.status_code}")

    @property
    def config(self):
        """Get the config object for reading values."""
        if self.__initialized is False:
            print("Config is not set yet, did you run init_manager?")
        return self.__config

    @property
    def connectors(self):
        """Get the connectors list."""
        if self.__initialized is False:
            print("Connectors are not set yet, did you run init_manager?")
        return self.__connectors

    @property
    def array_connectors(self):
        """Get the array connectors list."""
        if self.__initialized is False:
            print("Array connectors are not set yet, did you run init_manager?")
        return self.__array_connectors

    @property
    def service_id(self) -> Optional[str]:
        """Get the service ID of the current service."""
        # Check if the private attribute exists (handles name mangling)
        if not hasattr(self, '_ConfigManager__id'):
            return None
        return self.__id

    def getStackIdForConnector(self, config_id: str) -> Union[str, List[str], None]:
        """
        Return the stack ID(s) associated with a given configuration ID.

        Args:
            config_id (str): The configuration ID to find the stack ID(s) for.

        Returns:
            Union[str, List[str], None]: The stack ID, list of stack IDs if array connector, or None.
        """
        connector: ConnectorSchema
        for connector in self.__connectors:
            if connector.config_id == config_id:
                return connector.stack_id

        arrayconnector: ArrayConnectorSchema
        if self.__array_connectors:
            for arrayconnector in self.__array_connectors:
                if arrayconnector.config_id == config_id:
                    return arrayconnector.stack_ids
                    
        return None

    def getNamespaceForConnector(self, config_id: str) -> Optional[str]:
        """
        Determine the namespace associated with a given configuration ID based on its type.

        Args:
            config_id (str): The configuration ID to find the namespace for.

        Returns:
            str | None: The namespace if found, otherwise None.
        """
        # Check regular connectors
        for connector in self.__connectors:
            if connector.config_id == config_id:
                return get_namespace_for_type(connector.type)
            
        # Check array connectors
        if self.__array_connectors:
            for connector in self.__array_connectors:
                if connector.config_id == config_id:
                    return get_namespace_for_type(connector.type)
        
        return None

    def getUrlForConnector(self, config_id: str) -> Union[str, List[str], None]:
        """
        Construct and return the URL(s) associated with a given configuration ID.
        In local development mode, checks for environment variables in the format:
        {config_id}_URL for regular connectors
        {config_id}_0_URL, {config_id}_1_URL, etc. for array connectors
        
        Args:
            config_id (str): The configuration ID to find the URL(s) for.

        Returns:
            Union[str, List[str], None]: The constructed URL, list of URLs if array connector, or None.
        """
        if config.LOCAL_DEV:
            # Check for array connector environment variables
            urls = []
            index = 0
            while True and index < 10:  # Safety limit to prevent infinite loops
                url_var = f"{config_id}_{index}_URL"
                url = os.environ.get(url_var)
                if url is None:
                    break
                urls.append(re.sub(r"/$", "", url))  # Remove trailing slash if exists
                index += 1
                
            # If we found array connector URLs, return them
            if urls:
                return urls
                
            # Otherwise, check for regular connector URL
            url = os.environ.get(f"{config_id}_URL")
            if url is not None:
                return re.sub(r"/$", "", url)  # Remove trailing slash if exists
                
            # If no environment variables are defined, fall back to standard behavior
                
        # Standard behavior for non-local or when no environment variables are set
        stack_ids = self.getStackIdForConnector(config_id)
        if stack_ids is None:
            return None

        namespace = self.getNamespaceForConnector(config_id)
        if namespace is None:
            return None

        def construct_url(stack_id: str) -> str:
            base_url = f"http://{stack_id}.{namespace}.svc.cluster.local:8080"
            if namespace == "databases":
                base_url = f"{stack_id}-rw.{namespace}.svc.cluster.local"
            return base_url

        # Handle both single stack_id and array of stack_ids
        if isinstance(stack_ids, list):
            return [construct_url(stack_id) for stack_id in stack_ids]
        return construct_url(stack_ids)

    def getUrlAndNameForConnector(self, config_id: str) -> Union[Dict[str, str], List[Dict[str, str]], None]:
        """
        Construct and return the URL(s) and name(s) associated with a given configuration ID.

        Args:
            config_id (str): The configuration ID to find the URL(s) for.

        Returns:
            Union[Dict[str, str], List[Dict[str, str]], None]: The constructed URL and Name, list of URLs if array connector, or None.
            If the display name is found, returns a dictionary with URL and name.
            
        Example:
            url_base = configManager.getUrlForConnector(LLM_CONFIG_ID, include_name=True)
            # For regular connector:
            # Returns: {"url": "http://stack-id.namespace.svc.cluster.local:8080", "name": "test name"}
            # For array connector:
            # Returns: [{"url": "http://stack-id1.namespace.svc.cluster.local:8080", "name": "test name 1"},
            #           {"url": "http://stack-id2.namespace.svc.cluster.local:8080", "name": "test name 2"}]
        """
        stack_ids = self.getStackIdForConnector(config_id)
        if stack_ids is None:
            return None

        namespace = self.getNamespaceForConnector(config_id)
        if namespace is None:
            return None

        def construct_url_and_name(stack_id: str, deployment_name: Optional[str]) -> Dict[str, str]:
            if namespace == "databases":
                url = f"{stack_id}-rw.{namespace}.svc.cluster.local"
            else:
                url = f"http://{stack_id}.{namespace}.svc.cluster.local:8080"
            return {"url": url, "name": deployment_name or stack_id}

        if isinstance(stack_ids, list):
            for arrayconnector in self.__array_connectors:
                if arrayconnector.config_id == config_id:
                    deployment_names = arrayconnector.deployment_names or [None] * len(arrayconnector.stack_ids)
                    result = []
                    for stack_id, deployment_name in zip(arrayconnector.stack_ids, deployment_names):
                        result.append(construct_url_and_name(stack_id, deployment_name))
                    return result
            return None

        else:
            for connector in self.__connectors:
                if connector.config_id == config_id:
                    return construct_url_and_name(connector.stack_id, connector.deployment_name)
            return None


class BaseToolConfig(BaseModel):
    name: str = "undefined"
    type: Literal["undefined"] = "undefined"